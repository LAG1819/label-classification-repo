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

from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, get_scheduler, create_optimizer
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datasets import Dataset, DatasetDict
import evaluate
import os 
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from ray import tune 
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import HyperBandScheduler,HyperBandForBOHB
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray import air


class GERBertClassifier(nn.Module):
  def __init__(self,checkpoint,num_labels): 
    super(GERBertClassifier,self).__init__() 
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
        

def load_data(text_col):
        df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\labeled_texts.feather")
        df = pd.read_feather(df_path)
        data = df.replace(np.nan, "",regex = False)
        data = data[:1000]
        data['text'] = data[text_col]
        
        train, validate, test = np.split(data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(data)), int(.8*len(data))])

        dataset = {}

        dataset['train'] = train[['LABEL','text']].to_dict('records')#.to_records(index=False)
        dataset['test'] = test[['LABEL','text']].to_dict('records')
        dataset['val'] = validate[['LABEL','text']].to_dict('records')
        
        return dataset

def preprocess_data(data, tokenizer):
    tokenized_data = {'train':[],'test':[],'val':[]}
    for set in data:
        for sample in data[str(set)]:
            tokenized_sent = tokenizer(sample["text"], truncation=True, max_length = 512)
            tokenized_sent.pop('token_type_ids')
            tokenized_sent['label'] = [sample["LABEL"]]
            tokenized_data[str(set)].append(tokenized_sent)
    return tokenized_data

def transform_train_data(tokenized_data, data_collator, config_batch_size):
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

def transform_eval_data(tokenized_data,data_collator,batch_size):
    data = DatasetDict({
    'val': Dataset.from_list(tokenized_data['val'])
    })
    data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
    
    eval_dataloader = DataLoader(
        data["val"], batch_size=batch_size, collate_fn=data_collator
    )
    return eval_dataloader

def set_params(model, config_lr, len_train_data):   
    optimizer = AdamW(model.parameters(), lr=config_lr)
    num_epochs = 1
    num_training_steps = num_epochs * len_train_data
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    metric = evaluate.load("accuracy")

    return num_training_steps,num_epochs, metric, optimizer, lr_scheduler

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = GERBertClassifier(checkpoint="bert-base-german-cased",num_labels=9).to(device)

    train_test_val_set = load_data(config["col"])
    len_train_dl = len(train_test_val_set['train'])

    tokenized_train_test_set = preprocess_data(train_test_val_set, tokenizer)

    train_dl,test_dl = transform_train_data(tokenized_train_test_set, data_collator,config["batch_size"])
    
    num_training_steps,num_epochs, metric, optimizer, lr_scheduler = set_params(model, config['lr'], len_train_dl)

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
            acc = metric.compute()
        
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\pytorch_tuning_de\models_checkpoint\gerbert.pt"
        os.makedirs("gerbert", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict()), path)
        checkpoint = Checkpoint.from_directory("gerbert")
        session.report({"accuracy": acc['accuracy']}, checkpoint=checkpoint)#"loss": (val_loss / val_steps), 
    

def validate_model(text_col, best_results):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = GERBertClassifier(checkpoint="bert-base-german-cased",num_labels=9).to(device)
    
    train_test_val_set = load_data(text_col)
    tokenized_train_test_set = preprocess_data(train_test_val_set, tokenizer)
    eval_dl = transform_eval_data(tokenized_train_test_set,data_collator,best_results.config["batch_size"])

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
    

def predict(sentence, tokenizer, model, data_collator,batch_size,device):
    ag_labels = {2:"AUTONOMOUS",3:"CONNECTIVITY",4:"DIGITALISATION",5:"ELECTRIFICATION",6:"INDIVIDUALISATION",7:"SHARED",8:"SUSTAINABILITY"}
    tokenized_sent = tokenizer(sentence, truncation=True, max_length = 512)
    tokenized_sent.pop('token_type_ids')
    tokenized_sent['label'] = 0
    tokenized_data = {'val':[tokenized_sent]}
    train_dl, test_dl, val_dl = transform_eval_data(tokenized_data,data_collator,batch_size)
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)[0].item()
        label = ag_labels[prediction]
        print(prediction)
        print(label)

def run(lang ='de', col = 'text'):
    config_rand = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":tune.choice([2,4,6,8,16]),
        "lang":lang,
        "col":col
    } 
    ###Random Search###
    path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\pytorch_tuning_de"
    tuner_random = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 3}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples = 5,
        ),
        param_space=config_rand,
        run_config=air.RunConfig(local_dir=path, name="random_search_de")
    )
    result_rand = tuner_random.fit()
    best_results_rand = result_rand.get_best_result("accuracy", "max")
    
    config = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":tune.choice([2,4,6,8,16]),
        "lang":lang,
        "col":col
    }
    ###Bayesian Optimization Search###
    bayes = BayesOptSearch(random_search_steps=4)
    tuner_bayes = tune.Tuner(
        train_model,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            search_alg=bayes,
        ),
        run_config=air.RunConfig(stop={"training_iteration": 1},local_dir=path, name="bayes_search_de"),
        param_space=config,
    )
    result_bayes = tuner_bayes.fit()
    best_results_bayes = result_bayes.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_results_bayes.config))
    print("Best trial final validation loss: {}".format(best_results_bayes.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_results_bayes.metrics["accuracy"]))

    ###Hyberpban Bayesian Optimization###
    bohb = HyperBandForBOHB(metric = "accuracy", mode = "max")

    ###Hyperband Optimization###
    hyperband = HyperBandScheduler(metric = "accuracy", mode = "max")
    tuner_hyper = tune.Tuner(
        train_model,
        tune_config = tune.TuneConfig(
            num_samples = 10,
            scheduler = hyperband
        ),
        param_space = config,
        run_config=air.RunConfig(local_dir=path, name="hyperband_de")
    )
    
    result = tuner_hyper.fit()
    best_results = result.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_results.config))
    print("Best trial final validation loss: {}".format(best_results.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_results.metrics["accuracy"]))
    
    # validate_model(col, best_results)

    # self.predict("Connectivität ist digitale Vernetzung")
    print("FIN")


run()
    

