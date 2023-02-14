# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) 2023  Luisa-Sophie Gloger

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers import AutoTokenizer, BertTokenizer, DataCollatorWithPadding,AutoTokenizer,AutoModel,AutoConfig, get_scheduler, create_optimizer
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging 
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict
import evaluate
import os 
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from ray import tune 
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandScheduler,HyperBandForBOHB
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray import air


class BertClassifier(nn.Module):
  def __init__(self,checkpoint,num_labels): 
    super(BertClassifier,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)
        

def load_data(text_col:str,lang:str) -> dict:
    """Loads labeled dataset from files/04_classify

    Args:
        text_col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT. 
                        If TOPIC is choosen, the data had been labeled based on the previous extracted TOPICS of each text (for more info check src/cleans/topic_lda.py)
                        If URL_TEXT is choosen, the data had been labeled based on the previous cleaned texts (for more info check src/cleans/clean.py)
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)

    Returns:
        dict: Returns dictionary containing three datasets: 60% training, 20% testing and 20% validation dataset generated from the whole dataset. 
    """
    dir = r"files\04_classify\labeled_texts_"+lang+"_"+text_col+r".feather"
    df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],dir)
    df = pd.read_feather(df_path)
    data = df.replace(np.nan, "",regex = False)
    # data = data[:1000]
    data['text'] = data[text_col]
    
    train, validate, test = np.split(data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(data)), int(.8*len(data))])

    dataset = {}

    dataset['train'] = train[['LABEL','text']].to_dict('records')#.to_records(index=False)
    dataset['test'] = test[['LABEL','text']].to_dict('records')
    dataset['val'] = validate[['LABEL','text']].to_dict('records')
    
    return dataset

def preprocess_data(data:dict, tokenizer) -> dict:
    tokenized_data = {'train':[],'test':[],'val':[]}
    for set in data:
        for sample in data[str(set)]:
            tokenized_sent = tokenizer(sample["text"], truncation=True, max_length = 512)
            tokenized_sent.pop('token_type_ids')
            tokenized_sent['label'] = [sample["LABEL"]]
            tokenized_data[str(set)].append(tokenized_sent)
    return tokenized_data

def transform_train_data(tokenized_data:dict, data_collator:DataCollatorWithPadding, config_batch_size:int):
    data = DatasetDict({
    'train': Dataset.from_list(tokenized_data['train']),
    'test': Dataset.from_list(tokenized_data['test'])
    })
    data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
    
    train_dataloader = DataLoader(
        data["train"], shuffle=True, batch_size=config_batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        data["test"], batch_size=config_batch_size, collate_fn=data_collator
    )
    return train_dataloader, test_dataloader

def transform_eval_data(tokenized_data:dict,data_collator:DataCollatorWithPadding,batch_size:int):
    data = DatasetDict({
    'val': Dataset.from_list(tokenized_data['val'])
    })
    data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
    
    eval_dataloader = DataLoader(
        data["val"], batch_size=batch_size, collate_fn=data_collator
    )
    return eval_dataloader

def set_params(model, config_lr, config_epoch,len_train_data):   
    optimizer = AdamW(model.parameters(), lr=config_lr)
    num_epochs = config_epoch
    num_training_steps = num_epochs * len_train_data
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    metric = evaluate.load("accuracy")
    metric2 = evaluate.load("roc_auc","multilabel")

    return num_training_steps,num_epochs, metric, optimizer, lr_scheduler,metric2

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['lang'] == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-uncased")
        model = BertClassifier(checkpoint="bert-base-german-uncased",num_labels=7).to(device)
    elif config['lang'] == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertClassifier(checkpoint='bert-base-uncased',num_labels=7).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train_test_val_set = load_data(config["col"],config['lang'])
    # len_train_dl = len(train_test_val_set['train'])

    # tokenized_train_test_set = preprocess_data(train_test_val_set, tokenizer)

    # train_dl,test_dl = transform_train_data(tokenized_train_test_set, data_collator,config["batch_size"])
    train_dl,test_dl = transform_train_data(config["tokenized_train_test_set"], data_collator,config["batch_size"])
    num_training_steps,num_epochs, metric, optimizer, lr_scheduler,metric2 = set_params(model, config['lr'],config['epoch'], config['len_train_dl'])

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            #metric2.add_batch(predictions=predictions, references=batch["labels"])
            acc = metric.compute()
            #rocauc = metric2.compute()
        
        os.makedirs("bert", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict()), "bert/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("bert")
        session.report({"accuracy": acc['accuracy']}, checkpoint=checkpoint)#, "roc_auc":rocauc["roc_auc"]

def random_search(lang:str, col:str, path:str,tokenized_train_test_set_fold,len_train_dl,num_samples = 1):
    config_rand = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":tune.choice([2,4,6,8,16]),
        "epoch":tune.choice([1,3,5,7,10]),
        "lang":lang,
        "col":col,
        "tokenized_train_test_set":tokenized_train_test_set_fold,
        'len_train_dl': len_train_dl
    } 
    tuner_random = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples = num_samples,
        ),
        param_space=config_rand,
        run_config=air.RunConfig(local_dir=path, name="random_search_"+lang)
    )
    result_rand = tuner_random.fit()
    best_results_rand = result_rand.get_best_result("accuracy", "max")
    print("[RAND]Best trial config: {}".format(best_results_rand.config))
    print("[RAND]Best trial final validation accuracy: {}".format(best_results_rand.metrics["accuracy"]))
    return best_results_rand

def bohb(lang:str,col:str,path:str,tokenized_train_test_set_fold,len_train_dl,num_samples=1):
    config = {
        "lr":tune.loguniform(1e-4,1e-1),
        "epoch":tune.choice([1,3,5,7,10]),
        "batch_size":tune.choice([2,4,6,8,16]),
        "lang":lang,
        "col":col,
        "tokenized_train_test_set":tokenized_train_test_set_fold,
        'len_train_dl': len_train_dl
    }
    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1,
        reduction_factor=4,
        stop_last_trials=False,)
    bohb_search = TuneBOHB()
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=2)
    tuner_bayes = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=bohb,
            search_alg=bohb_search,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(stop={"training_iteration": 1},local_dir=path, name="bohb_search_"+lang),
        param_space=config,
    )
    result_bayes = tuner_bayes.fit()
    best_results_bayes = result_bayes.get_best_result("accuracy", "max")

    print("[BOHB]Best trial config: {}".format(best_results_bayes.config))
    print("[BOHB]Best trial final validation accuracy: {}".format(best_results_bayes.metrics["accuracy"]))
    return best_results_bayes

def hyperband(lang:str, col:str, path:str, tokenized_train_test_set_fold,len_train_dl,num_samples=1):
    config = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":tune.choice([2,4,6,8,16]),
        "epoch":tune.choice([1,3,5,7,10]),
        "lang":lang,
        "col":col,
        "tokenized_train_test_set":tokenized_train_test_set_fold,
        'len_train_dl':len_train_dl
    }
    hyperband = HyperBandScheduler(metric = "accuracy", mode = "max")
    tuner_hyper = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 1}
        ),
        tune_config = tune.TuneConfig(
            num_samples = num_samples,
            scheduler = hyperband
        ),
        param_space = config,
        run_config=air.RunConfig(local_dir=path, name="hyperband_"+lang)
    )
    
    result = tuner_hyper.fit()
    best_results = result.get_best_result("accuracy", "max")
    
    print("[HYPER]Best trial config: {}".format(best_results.config))
    print("[HYPER]Best trial final validation accuracy: {}".format(best_results.metrics["accuracy"]))
    return best_results
    
    
def validate_model(text_col:str,lang:str, best_results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lang == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-uncased")
        model = BertClassifier(checkpoint="bert-base-german-uncased",num_labels=7).to(device)
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertClassifier(checkpoint='bert-base-uncased',num_labels=7).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoint_path = os.path.join(best_results["Checkpoint"].to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)
    
    train_test_val_set = load_data(text_col)
    tokenized_train_test_set = preprocess_data(train_test_val_set, tokenizer)
    eval_dl = transform_eval_data(tokenized_train_test_set,data_collator,best_results["Configuration"]["batch_size"])

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print(metric.compute())

    
    path_to_save_model = str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\trained_model_'+lang+'_'+text_col+".pth"
    torch.save(model, path_to_save_model)
    

def predict(sentence:str, lang:str,batch_size, text_col = 'TOPIC'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    path_to_load_model = str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\trained_model_'+lang+'_'+text_col+".pth"
    model = torch.load(path_to_load_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if lang == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    tokenized_sent = tokenizer(sentence, truncation=True, max_length = 512)
    tokenized_sent.pop('token_type_ids')
    tokenized_sent['label'] = 0
    tokenized_data = {'val':[tokenized_sent]}
    train_dl, test_dl, val_dl = transform_eval_data(tokenized_data,data_collator,batch_size)
    

    ag_labels = {0:"AUTONOMOUS",1:"CONNECTIVITY",2:"DIGITALISATION",3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"}
    
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)[0].item()
        label = ag_labels[prediction]
        print(prediction)
        print(label)

def run(lang:str, col:str):
    # Create logger and assign handler
    logger = logging.getLogger("Labeler")
    if (logger.hasHandlers()):
        logger.handlers.clear()

    handler  = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    filenames =  str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\pytorch_tuning_'+lang+r'\classifier_training_'+lang+r'.log'
    fh = logging.FileHandler(filename=filenames)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(fh)

    if lang == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-uncased")
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
    train_test_val_set = load_data(col,lang)
    len_train_dl = len(train_test_val_set['train'])

    tokenized_train_test_set = preprocess_data(train_test_val_set, tokenizer)
    tokenized_data = tokenized_train_test_set['train']+tokenized_train_test_set['test']
    tokenized_data = pd.DataFrame(tokenized_data)

    path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\pytorch_tuning_"+lang
    num_samples_per_tune = 1
    best_results = []
    try:
        logger.info(f"Classification tuning started with Language {lang}, Text-Column: {col} and Data source file: 'files\04_classify\labeled_texts_{lang}_{col}.feather'.")
        #K-Fold Cross Validation
        for k in range(2,10):
            k_fold = KFold(n_splits = k,shuffle = True, random_state = 12)
            i = 1
            tokenized_train_test_set_fold = {}
            for split in k_fold.split(tokenized_data):
                logger.info(f"Training of {k}-Fold Cross-Validation with Trainingsplit {i} started.")
                tokenized_train_test_set_fold['train'] = tokenized_data.iloc[split[0]].to_dict('records')
                tokenized_train_test_set_fold['test'] = tokenized_data.iloc[split[1]].to_dict('records')
                len_train_dl = tokenized_data.iloc[split[0]].shape[0]

                ###Random Search###
                rand_best_results = random_search(lang,col,path,tokenized_train_test_set_fold,len_train_dl,num_samples_per_tune)
                best_results.append({"Type":"Random Search","Accuracy": rand_best_results.metrics['accuracy'],"Configuration":rand_best_results.config,"Log_Dir":rand_best_results.log_dir, "Checkpoint":rand_best_results.checkpoint})
                logger.info(f"[Random Search]Best Model with Accuracy: {rand_best_results.metrics['accuracy']} and Configuration:{rand_best_results.config} reached. Checkpoint: {rand_best_results.checkpoint}")
                
                ###Hyberpban Bayesian Optimization###
                bohb_best_results = bohb(lang,col,path,tokenized_train_test_set_fold,len_train_dl,num_samples_per_tune)
                best_results.append({"Type":"BOHB","Accuracy":bohb_best_results.metrics['accuracy'],"Configuration":bohb_best_results.config,"Log_Dir": bohb_best_results.log_dir, "Checkpoint":bohb_best_results.checkpoint})
                logger.info(f"[BOHB]Best Model with Accuracy: {bohb_best_results.metrics['accuracy']} and Configuration:{bohb_best_results.config} reached. Checkpoint: {bohb_best_results.checkpoint}")
                
                ####Hyperband Optimization###
                hyperband_best_results = hyperband(lang, col, path,tokenized_train_test_set_fold,len_train_dl,num_samples_per_tune)
                best_results.append({"Type":"Hyperband","Accuracy":hyperband_best_results.metrics['accuracy'],"Configuration":hyperband_best_results.config,"Log_Dir":hyperband_best_results.log_dir, "Checkpoint":hyperband_best_results.checkpoint})
                logger.info(f"[Hyperband]Best Model with Accuracy:{hyperband_best_results.metrics['accuracy']} and Configuration:{hyperband_best_results.config} reached. Checkpoint: {hyperband_best_results.checkpoint}")

    except KeyboardInterrupt:       
        logger.info("KeyboardInterrupt. Current best model will be validated and saved if better than (possible) existing model.")
    finally:
        best_results_df = pd.DataFrame(best_results).sort_values(by=['Accuracy'], ascending=[False])
        best_result_acc = best_results_df.to_dict('records')[0]["Accuracy"]
        best_result_config = best_results_df.to_dict('records')[0]["Configuration"]
        logger.info(f"Overall best Model with Accuracy:{best_result_acc} and Configuration:{best_result_config} reached")
        
        ####Test Model###
        validate_model(col,lang, best_results)
        print("Done")


#run(lang ='de', col = 'TOPIC')
#run(lang ='en', col = 'TOPIC')
#run(lang ='de', col = 'URL_TEXT')
#run(lang ='en', col = 'URL_TEXT')

# ###Test new sample sentence###
# predict("Connectivität ist digitale Vernetzung")
    

