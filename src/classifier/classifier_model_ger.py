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

class ClassifierBERT:
    def __init__(self,lang = 'de'):
        self.lang = lang
        self.text_col = 'text'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = GERBertClassifier(checkpoint="bert-base-german-cased",num_labels=9).to(self.device)
        self.run()

    def load_data(self):
            df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\labeled_texts.feather")
            df = pd.read_feather(df_path)
            data = df.replace(np.nan, "",regex = False)
            data = data[:1000]
            data['text'] = data[self.text_col]
            
            train, validate, test = np.split(data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(data)), int(.8*len(data))])

            dataset = {}

            dataset['train'] = train[['LABEL','text']].to_dict('records')#.to_records(index=False)
            dataset['test'] = test[['LABEL','text']].to_dict('records')
            dataset['val'] = validate[['LABEL','text']].to_dict('records')
            
            return dataset

    def preprocess_data(self,data):
        tokenized_data = {'train':[],'test':[],'val':[]}
        for set in data:
            for sample in data[str(set)]:
                tokenized_sent = self.tokenizer(sample["text"], truncation=True, max_length = 512)
                tokenized_sent.pop('token_type_ids')
                tokenized_sent['label'] = [sample["LABEL"]]
                tokenized_data[str(set)].append(tokenized_sent)
        return tokenized_data

    def transform_data(self,tokenized_data):
        data = DatasetDict({
        'train': Dataset.from_list(tokenized_data['train']),
        'test': Dataset.from_list(tokenized_data['test']),
        'val': Dataset.from_list(tokenized_data['val'])
        })
        data.set_format('torch', columns = ['input_ids', 'attention_mask', 'label'])
        
        train_dataloader = DataLoader(
            data["train"], shuffle=True, batch_size=2, collate_fn=self.data_collator
        )
        test_dataloader = DataLoader(
            data["test"], batch_size=2, collate_fn=self.data_collator
        )
        eval_dataloader = DataLoader(
            data["val"], batch_size=2, collate_fn=self.data_collator
        )
        return train_dataloader, test_dataloader, eval_dataloader

    def set_params(self,len_train_data):
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
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

    def train_model(self,num_training_steps,num_epochs, metric, optimizer, lr_scheduler,train_dl,test_dl):
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.model.eval()
            for batch in test_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])  
            print(metric.compute())

    def validate_model(self,metric, eval_dl):
        self.model.eval()
        for batch in eval_dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        print(metric.compute())
        

    def predict(self, sentence):
        ag_labels = {2:"AUTONOMOUS",3:"CONNECTIVITY",4:"DIGITALISATION",5:"ELECTRIFICATION",6:"INDIVIDUALISATION",7:"SHARED",8:"SUSTAINABILITY"}
        tokenized_sent = self.tokenizer(sentence, truncation=True, max_length = 512)
        tokenized_sent.pop('token_type_ids')
        tokenized_sent['label'] = 0
        tokenized_data = {'train':[tokenized_sent],'test':[tokenized_sent],'val':[tokenized_sent]}
        train_dl, test_dl, val_dl = self.transform_data(tokenized_data)
        for batch in train_dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)[0].item()
            label = ag_labels[prediction]
            print(prediction)
            print(label)

    def run(self):
        train_test_val_set = self.load_data()
        len_train_data = len(train_test_val_set['train'])

        tokenized_train_test_set = self.preprocess_data(train_test_val_set)
        train_dl,test_dl,val_dl = self.transform_data(tokenized_train_test_set)

       
        num_training_steps,num_epochs, metric, optimizer, lr_scheduler = self.set_params(len_train_data)

        self.train_model(num_training_steps,num_epochs, metric, optimizer, lr_scheduler,train_dl, test_dl)
        self.validate_model(metric,val_dl)

        self.predict("Connectivität ist digitale Vernetzung")
        print("FIN")


gerbert = ClassifierBERT()
    

