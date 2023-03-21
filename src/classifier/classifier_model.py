# <BERT-based text classification. Classification modell is trained seperately on topics or texts in englisch or german for comparison reasons. This is process step 5.>
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
from datasets import Dataset, DatasetDict
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW 
import pandas as pd
import numpy as np
import logging 
from sklearn.model_selection import KFold
import os 
import ray
from ray import tune 
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandScheduler,HyperBandForBOHB
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray import air
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
from functools import partial
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
N_CLASS = 7 #Number of classes to train classifier. Defaults to 7
NUM_WORKERS = 1 #Number of workers for DataLoader. Defaults to 1.
NUM_CPU = 2 #Number of cpu to use. Defaults to 2.
NUM_GPU = 1 #Number of gpu to use. Defaults to 1.
NUM_TRIALS = 1 #Number of samples to train. Defaults to 1.
NUM_TRIAL_ITER = 3 #Number of iterations a samples trains. Defaults to 3.
PIN_MEM = True #Boolean to set memory pin. If true data dirctly to gpu. Defaults to True.
BATCH_SIZE = tune.choice([2,4,6,8]) #Size of batches to split data. Defaults to tune.choice with option 2,4,6, and 8.

#pip install datasets transformers numpy pandas evaluate scikit-learn hpbandster "ray[default]" "ray[tune]" "ray[air]"
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

class BertClassifier(nn.Module):
    """Customised Bert Classifier. The Class takes a pretrained Model as input and adds a Classifier Layer on top.
    Pretrained Modell Bert-based (german): bert-base-german-dbmdz-uncased
    Pretrained Modell Bert-based (english): bert-base-uncased
    Args:
        nn (_type_): Neurol Network Module from Pytorch
    """
    def __init__(self,checkpoint,num_labels):
        """Initialisation of a given pytorch model.

        Args:
            checkpoint (_type_): Checkpoint to load of a (pretrained) pytorch model for text classification.
            num_labels (_type_): Given number of classes the classification layer has, that will be added on top of the (pretrained) pytorch model.
        """
        super(BertClassifier,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 
        self.classifier = nn.Linear(768,num_labels) # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        """Obligatory forward function of a pytorch nn model. Takes ids, attention maksk AND labels as input. Those are generated of given data in prerocess_data function.

        Args:
            input_ids (_type_, optional): _description_. Defaults to None.
            attention_mask (_type_, optional): _description_. Defaults to None.
            labels (_type_, optional): _description_. Defaults to None.

        Returns:
            TokenClassifierOutput: Returns Base class for outputs of token classification models.
        """
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

class ExperimentTerminationReporter(CLIReporter):
    """Helper class for ray tune. Manages reporting while tuning with selected hyperparameter optimization techniques. 
    Reports only after whole experiment of ray tune finished.

    Args:
        CLIReporter (_type_): Takes obligatory CLIReporter of ray as input to manage reporting.
    """
    def should_report(self, trials, done=False):
        """Reports only on experiment termination.

        Args:
            trials (_type_): trial objects containing status and other meta information about each trial of current tuning.
            done (bool, optional): Status of experiment. Returns True if experiment finished. Defaults to False.

        Returns:
           bool: Returns status of experiment.
        """
        return done

class TrialTerminationReporter(CLIReporter):
    """Helper class for ray tune. Manages reporting while tuning with selected hyperparameter optimization techniques. 
    Reports only after one trial in experiment of ray tune finished.

    Args:
        CLIReporter (_type_): Takes obligatory CLIReporter of ray as input to manage reporting.
    """
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination. Checks number of terminated trials and updates them.

        Args:
            trials (_type_): trial objects containing status and other meta information about each trial of current tuning.
            done (bool, optional): Status of experiment. Returns True if experiment finished. Defaults to False.

        Returns:
            bool: Returns status of experiment.
        """
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def _load_data(text_col:str,dir:str) -> dict:
    """Loads labeled dataset from files/04_classify

    Args:
        text_col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT. 
                        If TOPIC is choosen, the classifier will be trained on the extracted TOPICS of the cleaned text data (for more info check src/cleans/topic_lda.py)
                        If URL_TEXT is choosen, the classifier will be trained on on the cleaned texts (for more info check src/cleans/clean.py)
        dir (str): Selected path to the desired labeled data to train the classifier with. Defaults to r"files\04_classify\labeled_texts_"+lang+"_"+text_col+".feather" through run().

    Returns:
        dict: Returns dictionary containing three datasets: 60% training, 20% testing and 20% validation dataset generated from the whole dataset. 
    """
    _df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],dir)
    _df = pd.read_feather(_df_path)
    _data = _df.replace(np.nan, "",regex = False)
    # data = data[:100]
    _data['text'] = _data[text_col]
    
    _train, _validate, _test = np.split(_data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(_data)), int(.8*len(_data))])

    _dataset = {}

    _dataset['train'] = _train[['LABEL','text']].to_dict('records')
    _dataset['test'] = _test[['LABEL','text']].to_dict('records')
    _dataset['val'] = _validate[['LABEL','text']].to_dict('records')
    
    return _dataset

def _preprocess_data(data:dict, tokenizer) -> dict:
    """Prepares data for text classification:
            texts will be tokenized with corresponding tokenizer of loaded pytorch model.
            labels will be specifically set as key,value pair based on given label per sample.
    Args:
        data (dict): Dictionary containing text data. Contains trainset,texts, testset,texts and validationset,texts as key,value pair.
        tokenizer (_type_): Corresponding tokenizer of loaded pytorch model. Needs to be loaded as well as pytorch model. 

    Returns:
        dict: Returns dictionary containing preprocessd train,test and validation data. 
        Contains trainset,label, input_ids, attention_mask and testset,label, input_ids, attention_mask and validationset,label, input_ids, attention_mask as key, value pair.
    """
    _tokenized_data = {'train':[],'test':[],'val':[]}
    for set in data:
        for _sample in data[str(set)]:
            _tokenized_sent = tokenizer(_sample["text"], truncation=True, max_length = 512)
            _tokenized_sent.pop('token_type_ids')
            _tokenized_sent['label'] = [_sample["LABEL"]]
            _tokenized_data[str(set)].append(_tokenized_sent)
    return _tokenized_data

def _transform_train_data(tokenized_data:dict, data_collator:DataCollatorWithPadding, config_batch_size:int):
    """Transforms preprocessed trainset and testset with pytorch DataLoader. Adds weighting to data with WeightedRandomSampler to neutrolize unbalanced classes.

    Args:
        tokenized_data (dict): Dictionary containing preprocessd train and test and validation data. 
        Contains dataset,label, input_ids, attention_mask as key, value pair for each set.
        data_collator (DataCollatorWithPadding): Loaded DataCollator for applying Padding on texts.
        config_batch_size (int): Given batch size for DataLoader.

    Returns:
        DataLoader: Transofrmed train- and testset, ready to apply on model.
    """
    _data = DatasetDict({
    'train': Dataset.from_list(tokenized_data['train']),
    'test': Dataset.from_list(tokenized_data['test'])
    })
    _data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
    
    #Weight Classes trainset
    _train_labels_unique, _counts = np.unique(_data['train']['label'], return_counts= True)
    _train_weights_per_class = {}#[sum(counts) / cl for cl in counts]
    for f,cl in enumerate(_counts):
        _train_weights_per_class[_train_labels_unique[f]] = sum(_counts) / cl
    _train_weights = [_train_weights_per_class[l.item()] for l in _data['train']['label']]
    _train_sampler = WeightedRandomSampler(_train_weights, len(_data['train']['label']))

    ##Weight Classes testset
    _test_labels_unique, _counts = np.unique(_data['test']['label'], return_counts= True)
    _test_weights_per_class = {} #[sum(counts) / cl for cl in counts]
    for p,cl in enumerate(_counts):
        _test_weights_per_class[_test_labels_unique[p]] = sum(_counts) / cl
    _test_weights = [_test_weights_per_class[l.item()] for l in _data['test']['label']]
    _test_sampler = WeightedRandomSampler(_test_weights, len(_data['test']['label']))

    _train_dataloader = DataLoader(
        _data["train"], sampler = _train_sampler, batch_size=config_batch_size, collate_fn=data_collator, num_workers= NUM_WORKERS ,pin_memory=PIN_MEM
    )
    _test_dataloader = DataLoader(
        _data["test"], sampler = _test_sampler, batch_size=config_batch_size, collate_fn=data_collator, num_workers=NUM_WORKERS ,pin_memory=PIN_MEM
    )
    return _train_dataloader, _test_dataloader

def _transform_eval_data(tokenized_data:dict,data_collator:DataCollatorWithPadding,batch_size:int):
    """Transforms preprocessed validationset with pytorch DataLoader. Adds weighting to data with WeightedRandomSampler to neutrolize unbalanced classes.

    Args:
        tokenized_data (dict): Dictionary containing preprocessd train and test and validation data. 
        Contains dataset,label, input_ids, attention_mask as key, value pair for each set.
        data_collator (DataCollatorWithPadding): Loaded DataCollator for applying Padding on texts.
        config_batch_size (int): Given batch size for DataLoader.

    Returns:
        DataLoader: Transofrmed validationset, ready to apply on trained model.
    """
    _data = DatasetDict({
    'val': Dataset.from_list(tokenized_data['val'])
    })
    _data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])

    ##Weight Classes valtset
    _val_labels_unique, _counts = np.unique(_data['val']['label'], return_counts= True)
    _val_weights_per_class = {}#[sum(counts) / cl for cl in counts]
    for p,cl in enumerate(_counts):
        _val_weights_per_class[_val_labels_unique[p]] = sum(_counts) / cl
    _val_weights = [_val_weights_per_class[l.item()] for l in _data['val']['label']]
    _val_sampler = WeightedRandomSampler(_val_weights, len(_data['val']['label']))
    
    _eval_dataloader = DataLoader(
        _data["val"], batch_size=batch_size, collate_fn=data_collator, num_workers= NUM_WORKERS ,pin_memory=PIN_MEM
    )
    return _eval_dataloader

def _set_params(model:BertClassifier, config_lr:float, config_epoch:int,len_train_data:int):
    """Generate parameters and metrics for model training and evaluation.

    Args:
        model (BertClassifier): Loaded model for classification.
        config_lr (float): Learning rate.
        config_epoch (int): Number of epochs.
        len_train_data (int): Length of training data to determine number of training steps.

    Returns:
        parameters: Returns generated parameters and dedicted metrics for evaluation.
    """
    optimizer = AdamW(model.parameters(), lr=config_lr)
    _num_epochs = config_epoch
    _num_training_steps = _num_epochs * len_train_data
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=_num_training_steps,
    )

    #load metrics from hugging face evaluate
    accuracy = evaluate.load('accuracy')
    f1_mi = evaluate.load('f1')
    f1_ma = evaluate.load('f1')
    precision_mi = evaluate.load('precision')
    precision_ma = evaluate.load('precision')
    recall = evaluate.load('recall')
    # roc_auc = evaluate.load("roc_auc","multiclass")
    mcc = evaluate.load("matthews_correlation")

    return _num_training_steps,_num_epochs, optimizer, lr_scheduler,accuracy,f1_mi,f1_ma,precision_mi,precision_ma,recall, mcc

def _train_model(config:dict, data_dir:str, lang:str, col:str):
    """Training function of model. Follows the usual procedure consisting training and evaluation of the model with training and testing set.

    Args:
        config (dict): Configuration of model containing parameters and metrics.
        data_dir (str): Path to dataset to train with.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #select bert model for dedicated language
    if lang == 'de': #config['lang']
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-dbmdz-uncased")
        model = BertClassifier(checkpoint="bert-base-german-dbmdz-uncased",num_labels=N_CLASS)
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertClassifier(checkpoint='bert-base-uncased',num_labels=N_CLASS)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model.to(device)

    #load and preprocess data 
    loaded_data = _load_data(col, dir=data_dir)
    data = _preprocess_data(loaded_data, tokenizer)
    len_train_dl = len(data['train'])

    #transform loaded data
    train_dl,test_dl = _transform_train_data(data, data_collator, config["batch_size"])
    num_training_steps,num_epochs, optimizer, lr_scheduler,accuracy,f1_mi,f1_ma,pr_mi,pr_ma,recall, mcc = _set_params(model, config['lr'],config['epoch'], len_train_dl)

    # load last checkpoint if model training hab been interrupted
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    #train and evaluation of model 
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
            
            #references is list of list tensor -> convert to single tensor list
            # roc_auc.add_batch(prediction_scores=predictions, references=batch["labels"])#multi_class ='ovr'
            
            mcc.add_batch(predictions=predictions, references=batch["labels"])
            accuracy.add_batch(predictions=predictions, references=batch["labels"])
            f1_mi.add_batch(predictions=predictions, references=batch["labels"])
            f1_ma.add_batch(predictions=predictions, references=batch["labels"])
            pr_mi.add_batch(predictions=predictions, references=batch["labels"])
            pr_ma.add_batch(predictions=predictions, references=batch["labels"])
            recall.add_batch(predictions=predictions, references=batch["labels"])
            # metric.add_batch(predictions=predictions, references=batch["labels"], average = None)
                
            mcc_metrics = mcc.compute(average = 'macro')   
            acc_metric = accuracy.compute()
            f1_metric_micro = f1_mi.compute(labels = [0,1,2,3,4,5,6], average='micro')
            f1_metric_macro = f1_ma.compute(labels = [0,1,2,3,4,5,6], average='macro')
            precision_metric_micro = pr_mi.compute(labels = [0,1,2,3,4,5,6], average='micro', zero_division = 0)
            precision_metric_macro = pr_ma.compute(labels = [0,1,2,3,4,5,6], average='macro', zero_division = 0)
            recall_metric = recall.compute(labels = [0,1,2,3,4,5,6], average='macro', zero_division = 0)             
        
        os.makedirs("bert", exist_ok=True)
        torch.save((model.state_dict(), optimizer.state_dict()), "bert/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("bert")
        #roc_auc_metrics['roc_auc']
        session.report({"accuracy": acc_metric['accuracy'], "precisionMicro":precision_metric_micro['precision'], "precisionMacro":precision_metric_macro['precision'],\
                        "f1Micro": f1_metric_micro['f1'],"f1Macro": f1_metric_macro['f1'],"recall":recall_metric['recall'],\
                              "mcc":mcc_metrics['matthews_correlation']}, checkpoint=checkpoint)

def random_search(lang:str, col:str, path:str,data_path:str):
    """Hyperparameter Optimization techinique of Random Search using ray.tune. 

    Args:
        lang (str): Unicode of language to train model with. It can be choosen between de (german) and en (englisch).
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        data_path (str): Path to save experiment.

    Returns:
        Result: Returns metainformation and metrics of best trial.
    """
    config_rand = {
            "lr":tune.loguniform(1e-4,1e-1),
            "batch_size":BATCH_SIZE,
            "epoch":tune.choice([3,5,7,10]),
            # "lang":lang,
            # "col":col
        }
    if lang == 'en':
        config_rand["batch_size"] = tune.choice([2])
            
    tuner_random = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(_train_model, data_dir = data_path, lang = lang, col = col),
            resources={"cpu": NUM_CPU, "gpu":NUM_GPU}
        ),
        # tune.with_resources(
        #     tune.with_parameters(_train_model, data = tokenized_train_test_set_fold),
        #     resources={"cpu": num_cpu, "gpu":num_gpu}
        # ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples = NUM_TRIALS,
        ),
        param_space=config_rand,
        run_config=air.RunConfig(local_dir=path, name="random_search_"+lang, stop={"training_iteration":  NUM_TRIAL_ITER})#,progress_reporter=TrialTerminationReporter())#, progress_reporter=TrialTerminationReporter() OR ExperimentTerminationReporter()
    )
    result_rand = tuner_random.fit()
    best_results_rand = result_rand.get_best_result("accuracy", "max")
    # print("[RAND]Best trial config: {}".format(best_results_rand.config))
    # print("[RAND]Best trial final validation accuracy: {}".format(best_results_rand.metrics["accuracy"]))
    return best_results_rand

def bohb(lang:str,col:str,path:str,data_path:str):
    """Hyperparameter Optimization techinique of combination of Bayesian optimization (BO) and Hyperband (HB) using ray.tune. 

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        data_path (str): Path to save experiment.

    Returns:
        Result: Returns metainformation and metrics of best trial.
    """
    config = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":BATCH_SIZE,
        "epoch":tune.choice([3,5,7,10]),
        # "lang":lang,
        # "col":col
    }
    if lang == 'en':
        config["batch_size"] = tune.choice([2])

    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1,
        reduction_factor=4,
        stop_last_trials=False,)
    bohb_search = TuneBOHB()
    bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=2)
    tuner_bayes = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(_train_model, data_dir = data_path, lang = lang, col = col),
            resources={"cpu": NUM_CPU, "gpu":NUM_GPU}
        ),
        # tune.with_resources(
        #     tune.with_parameters(_train_model, data_dir = data_path)#data = tokenized_train_test_set_fold),
        #     resources={"cpu": num_cpu, "gpu":num_gpu}
        # ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=bohb,
            search_alg=bohb_search,
            num_samples=NUM_TRIALS,
        ),
        run_config=air.RunConfig(local_dir=path, name="bohb_search_"+lang,stop={"training_iteration": NUM_TRIAL_ITER}),#,progress_reporter=TrialTerminationReporter()),
        param_space=config,
    )
    result_bayes = tuner_bayes.fit()
    best_results_bayes = result_bayes.get_best_result("accuracy", "max")

    # print("[BOHB]Best trial config: {}".format(best_results_bayes.config))
    # print("[BOHB]Best trial final validation accuracy: {}".format(best_results_bayes.metrics["accuracy"]))
    return best_results_bayes

def hyperband(lang:str, col:str, path:str, data_path:str):
    """Hyperparameter Optimization techinique of Hyperband (HB) using ray.tune. 

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        data_path (str): Path to save experiment.

    Returns:
        Result: Returns metainformation and metrics of best trial.
    """
    
    config = {
        "lr":tune.loguniform(1e-4,1e-1),
        "batch_size":BATCH_SIZE,
        "epoch":tune.choice([3,5,7,10]),
        # "lang":lang,
        # "col":col
    }
    if lang == 'en':
        config["batch_size"] = tune.choice([2])

    hyperband = HyperBandScheduler(metric = "accuracy", mode = "max")
    tuner_hyper = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(_train_model, data_dir = data_path, lang = lang, col = col),
            resources={"cpu": NUM_CPU, "gpu":NUM_GPU}
        ),
        tune_config = tune.TuneConfig(
            num_samples = NUM_TRIALS,
            scheduler = hyperband
        ),
        param_space = config,
        run_config=air.RunConfig(local_dir=path, name="hyperband_"+lang,stop={"training_iteration": NUM_TRIAL_ITER})#stop={"training_iteration": num_training_iter},progress_reporter=TrialTerminationReporter())
    )
    
    result = tuner_hyper.fit()
    best_results = result.get_best_result("accuracy", "max")
    
    # print("[HYPER]Best trial config: {}".format(best_results.config))
    # print("[HYPER]Best trial final validation accuracy: {}".format(best_results.metrics["accuracy"]))
    return best_results
    
    
def _validate_model(text_col:str,lang:str, data_path:str, best_result, model_path:str, model_name:str):
    """Validation function of model. Follows the usual procedure consisting testing of the optimized trained model with validation set.

    Args:
        text_col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        best_results (_type_): _description_
        model_path (str): Path of best trained model.
        model_name (str): Model name containing metainformation about language, texttype and random model number(unique).
    """
    torch.cuda.empty_cache()
    logger = logging.getLogger("Classification")
    print("Validate best found model and save it.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lang == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-dbmdz-uncased")
        model = BertClassifier(checkpoint="bert-base-german-dbmdz-uncased",num_labels=N_CLASS).to(device)
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertClassifier(checkpoint='bert-base-uncased',num_labels=N_CLASS).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoint_path = best_result["Checkpoint"]+ "\checkpoint.pt"
    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)
    
    train_test_val_set = _load_data(text_col, data_path)
    tokenized_train_test_set = _preprocess_data(train_test_val_set, tokenizer)
    eval_dl = _transform_eval_data(tokenized_train_test_set,data_collator,best_result["batch_size"])

    metric = evaluate.load("accuracy")

    model.eval()
    # with torch.no_grad():
    for batch in eval_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc= metric.compute()['accuracy']

    torch.save(model, model_path)
    logger.info(f"[Model {model_name}]Accuracy on validation set:{acc}. Saved to {model_path}")     
  
def _get_model_path(lang:str, text_col:str):
    """Get path of best trained model and model name to save best model to specfic path for loading.

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        text_col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.

    Returns:
        _type_: Returns model_name and model path as str.
    """
    i = _get_current_trial(lang,text_col)
    path_to_save_model = str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\trained_model_'+lang+'_'+text_col+"_"
    model_nr = str(i)+".pth"
    # while os.path.exists(path_to_save_model+model_nr):
    #     i += 1
    #     model_nr = str(i)+".pth"
    model_path = path_to_save_model+model_nr    
    model_name = 'trained_model_'+lang+'_'+text_col+"_"+model_nr
    return model_name, model_path

def _get_current_trial(lang:str,col:str)-> int:
    """Checks the number of trials based on evaluation data and sets trial number based on exististing number of trials.

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.

    Returns:
        int: Returns number of current trial.
    """
    
    t_path = r'models\classification\pytorch_tuning_'+lang+r'\results\eval_results_'+col+r'.feather'
    path = str(os.path.dirname(__file__)).split("src")[0]
    if os.path.exists(path+t_path):
        df_all = pd.read_feather(path+t_path)
        last_trial = df_all[['Trial']].sort_values(by=['Trial'], ascending=[False]).to_dict('records')[0]['Trial']
        trial = last_trial+1
    else:
        trial = 0
    return trial

def _save_current_result(lang:str, col:str, k:int,i:int,type:str, best_results):
    """Saves intermediate (best) evaluation results into a temporary file.

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        k (int): k of k-fold cross validation. Defaults to 0 as input if no k-fold cross validation is applied.
        i (int): i if split i in k-fold cross validation. Defaults to 0 as input if no k-fold cross validation is applied.
        type (str): Type of Optimization technique. It can be differnatiated between Random Search, Hyperband or BOHB.
        best_results (Result): Best Configuration of the Optimization.
    """
    if not os.path.exists(str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\pytorch_tuning_'+lang+r'\results'):
        os.makedirs(str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\pytorch_tuning_'+lang+r'\results')

    config = {"lr":best_results.config["lr"], 'batch_size':best_results.config['batch_size'], 'epoch':best_results.config['epoch']}
    # path = best_results.log_dir.__str__()
    checkpoint = best_results.checkpoint._local_path

    current_trial = _get_current_trial(lang,col)
    best_results_list = [{"Trial":current_trial,"Language":lang,"Text":col,"K-Fold":str(k),"Split":str(i), "Type":type,\
                                        "Accuracy": best_results.metrics['accuracy'],"PrecisionMicro":best_results.metrics["precisionMicro"],\
                                            "PrecisionMacro":best_results.metrics["precisionMacro"],"F1Micro": best_results.metrics["f1Micro"],\
                                                "F1Macro": best_results.metrics["f1Macro"],"RecallMacro":best_results.metrics['recall'],\
                                                "MCC":best_results.metrics["mcc"],"lr":best_results.config["lr"],'batch_size':best_results.config['batch_size'],\
                                                    'epoch':best_results.config['epoch'],"Log_Dir":best_results.log_dir.__str__(), "Checkpoint":str(checkpoint),\
                                                        'time_total_s': best_results.metrics['time_total_s'], 'training_iteration':best_results.metrics['training_iteration']}]
    df_new = pd.DataFrame(best_results_list)

    t_path = r'models\classification\pytorch_tuning_'+lang+r'\results\temp_eval_results_'+col+r'.feather'
    path = str(os.path.dirname(__file__)).split("src")[0]
    if os.path.exists(path+t_path):
        df_all = pd.read_feather(path+t_path)
        df_all_new = pd.concat([df_all,df_new])
    else:
        df_all_new = df_new
    
    df_all_new.reset_index(inplace = True, drop = True) 
    df_all_new.to_feather(path+t_path)

def _save_results(lang:str, col:str):
    """Saves all generated results in temporary evaluation sheet as evaluation results to dedicated result folder.

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
    """
    temp_t_path = r'models\classification\pytorch_tuning_'+lang+r'\results\temp_eval_results_'+col+r'.feather'
    t_path = r'models\classification\pytorch_tuning_'+lang+r'\results\eval_results_'+col+r'.feather'
    path = str(os.path.dirname(__file__)).split("src")[0]

    #get best trial
    best_results_df = pd.read_feather(path+temp_t_path)
    best_results_df.sort_values(by=['Accuracy'], ascending=[False], inplace=True)

    #check if target_path with previous trials already exists
    if os.path.exists(path+t_path):
        df_all = pd.read_feather(path+t_path)
        df_all_new = pd.concat([df_all,best_results_df])
    else:
        df_all_new = best_results_df

    #save dataset to target paths as feather
    #df_all_new = df_all_new[["Language", "Text", "K-Fold", "Split", "Type", "Accuracy", "Precision","F1", "Recall","Roc-auc", "Configuration"]]
    df_all_new.reset_index(inplace = True, drop = True) 
    df_all_new.to_feather(path+t_path)

    #remove temporary files
    os.remove(path+temp_t_path)

    return best_results_df.to_dict('records')[0]

def run(lang:str, col:str,data_path:str = None, list_of_hpo =[("RandomSearch",random_search),("Hyperband", hyperband),("BOHB", bohb)]):
    """Main function. Combines the training of the model with the different hyperparameter optimization techniques, 
    the validation of the best model and stores them including the evaluation results. Whole Run based on specified language and column

    Args:
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch).
        col (str): Selected Column on which the data had been labeled. It can be choosen between TOPIC or URL_TEXT.
        data_path (regexp, optional): Selected path to the desired labeled data to train the classifier with. Defaults to None.
        list_of_hpos (list, optional): List of hyperparameter optimization techniques to train classifier with. Defaults to [("RandomSearch",random_search),("Hyperband", hyperband),("BOHB", bohb)].
    """
    if not data_path:
        data_path =  r"files\04_classify\labeled_texts_"+lang+"_"+col+".feather"
    #generate temp path
    temp_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],r"models\classification\temp")
    ray.init(_temp_dir = temp_path)

    #generate training path if it does not exists
    path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\pytorch_tuning_"+lang
    if not os.path.exists(path):
        os.makedirs(path)

    # Create logger and assign handler
    logger = logging.getLogger("Classification")
    if (logger.hasHandlers()):
        logger.handlers.clear()

    handler  = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    __filenames =  str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\pytorch_tuning_'+lang+r'\classifier_training_'+lang+r'.log'
    fh = logging.FileHandler(filename=__filenames)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(fh)

    k = 0
    i = 0
    logger.info(f"################################################### Language: {lang}, Text-type: {col}, Labeled data: {data_path} ####################################################")     
    try:
        for hpo_name,hpo in list_of_hpo:
            torch.cuda.empty_cache()
            best_results = hpo(lang,col,path,data_path)
            _save_current_result(lang, col, k,i,hpo_name, best_results)
            logger.info(f"[{hpo_name}]Best Model with Accuracy: {best_results.metrics['accuracy']} and Configuration:{best_results.config} reached. Checkpoint: {best_results.checkpoint}")
   
    except RuntimeError as e:
        r= torch.cuda.memory_summary(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(r)
        torch.cuda.empty_cache()
        logger.info(f"[HPO]RuntimeError occurred:{e}")
    except Exception as f:
        logger.info(f"[HPO]Error occurred:{f}")
    except KeyboardInterrupt:
        logger.info(f"[HPO]KeyboardInterrupt.")
   
    finally:
        if(os.path.exists(str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\pytorch_tuning_'+lang+r'\results\temp_eval_results_'+col+r'.feather')):
            
            logger.info("Current best model will be validated and saved (if better than existing model).")
            
            #save evaluation data and identify best model and load for valdation
            model_name, model_path = _get_model_path(lang, col)
            best_result_record = _save_results(lang, col)
            lr = best_result_record["lr"]
            epoch = best_result_record["epoch"]
            logger.info(f"[Model {model_name}] Best Model with Accuracy:{best_result_record['Accuracy']} and Configuration: batch_size = {best_result_record['batch_size']}, lr = {lr}, epoch = {epoch}.")

            #validate best model on validation data###
            _validate_model(col,lang, data_path, best_result_record, model_path, model_name)

            ray.shutdown()
            torch.cuda.empty_cache()
            return
        else:
            ray.shutdown()
            torch.cuda.empty_cache()
            return
    
def predict(sentence:str, lang:str,text_col = 'TOPIC'):
    """Final Prediction Function. The trained model is loaded and predicts the class of the input sentence.

    Args:
        sentence (str): Input sentence to predict class of.
        lang (str): unicode of language to train model with. It can be choosen between de (german) and en (englisch)
        text_col (str, optional): Dedicated trained model. Currently selectable from 'TOPIC' or 'URL_TEXT' trained. Defaults to 'TOPIC' due to better evaluation results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_load_model = str(os.path.dirname(__file__)).split("src")[0] + r'models\classification\trained_model_'+lang+'_'+text_col+".pth"
    model = torch.load(path_to_load_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_size = 15

    if lang == 'de':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    elif lang == 'en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    tokenized_sent = tokenizer(sentence, truncation=True, max_length = 512)
    tokenized_sent.pop('token_type_ids')
    tokenized_sent['label'] = 0
    tokenized_data = {'val':[tokenized_sent]}
    train_dl = _transform_eval_data(tokenized_data,data_collator,batch_size)

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

#####Experiment 2#########
hpos =[("Hyperband", hyperband)]
# run(lang ='de', col = 'TOPIC', data_path = r"files\04_classify\Experiment2\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
run(lang ='en', col = 'TOPIC', data_path = r"files\04_classify\Experiment2\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)

# hpos = [("BOHB", bohb)]
# run(lang ='de', col = 'TOPIC', data_path = r"files\04_classify\Experiment2\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
#run(lang ='en', col = 'TOPIC', data_path = r"files\04_classify\Experiment2\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)

# hpos =[("Hyperband", hyperband)]
# run(lang ='de', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment2\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos,)
# run(lang ='en', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment2\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)
# hpos = [("BOHB", bohb)]
# run(lang ='de', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment2\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='en', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment2\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)


#####Experiment 1#########
# hpos =[("Hyperband", hyperband)]
# run(lang ='de', col = 'TOPIC', data_path = r"files\04_classify\Experiment1\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='de', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment1\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='en', col = 'TOPIC', data_path = r"files\04_classify\Experiment1\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='en', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment1\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)

# hpos = [("BOHB", bohb)]
# run(lang ='de', col = 'TOPIC', data_path = r"files\04_classify\Experiment1\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='de', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment1\labeled_texts_de_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='en', col = 'TOPIC', data_path = r"files\04_classify\Experiment1\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)
# run(lang ='en', col = 'URL_TEXT', data_path = r"files\04_classify\Experiment1\labeled_texts_en_TOPIC.feather",list_of_hpo=hpos)

# ###Test new sample sentence###
# predict("Connectivität ist digitale Vernetzung")

    