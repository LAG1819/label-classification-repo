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

from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import create_optimizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datasets import Dataset
from evaluate import load
import os
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

def load_data():
    df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\Output_texts_labeled.csv")
    df = pd.read_csv(df_path, header = 0, delimiter=",")
    data = df.replace(np.nan, "",regex = False)
    data['text'] = data['text'].apply(lambda row: row.replace("|","."))
    return generate_train_test(data)

def generate_train_test(data):
    target = data.pop('LABEL')
    features = data[['text']]

    # # set aside 20% of train and test data for evaluation
    train_ds, test_ds, train_label, test_label = train_test_split(features, target,
        test_size=0.2, shuffle = True, random_state = 8)

    # # Use the same function above for the validation set
    train_ds, val_ds, train_label, val_label = train_test_split(train_ds, train_label, 
        test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2
    
    print(train_ds.shape,val_ds.shape,test_ds.shape)

    train_ds['LABEL'] = train_label.to_frame()['LABEL'].tolist()
    val_ds['LABEL'] = val_label.to_frame()['LABEL'].tolist()
    test_ds['LABEL'] = test_label.to_frame()['LABEL'].tolist()

    dataset = {}

    dataset['train'] = train_ds[['LABEL','text']].to_dict('records')
    dataset['test'] = test_ds[['LABEL','text']].to_dict('records')

    return dataset, train_ds

def preprocess_data(data):
    tokenized_data = {'train':[],'test':[]}
    for set in data:
        for sample in data[str(set)]:
            tokenized_sent = tokenizer(sample["text"], truncation=True)
            tokenized_sent.pop('token_type_ids')
            tokenized_data[str(set)].append(tokenized_sent)

    return tokenized_data

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def transform_data_asTF(tokenized_data,data_collator):
    train_ds_encoded = Dataset.from_pandas(pd.DataFrame(tokenized_data['train']))
    test_ds_encoded = Dataset.from_pandas(pd.DataFrame(tokenized_data['test']))

    # print(train_ds_encoded.columns)
    # print(test_ds_encoded)

    train_ds_encoded.to_tf_dataset(
    columns= ['input_ids', 'token_type_ids', 'attention_mask'],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator
    )

    test_ds_encoded.to_tf_dataset(
    columns= ['input_ids', 'token_type_ids', 'attention_mask'],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator
    )

    return train_ds_encoded,test_ds_encoded

def transform_data_asTR(tokenized_data, data_collator):
    
    train_dataloader = DataLoader(
    tokenized_data["train"], shuffle=True, batch_size=2, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_data["test"], batch_size=2, collate_fn=data_collator
    )
    return train_dataloader, eval_dataloader

def set_params(len_train_data):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 2
    num_training_steps = num_epochs * len(len_train_data)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)
    metric = load("f1")

    return num_training_steps,num_epochs, metric, optimizer, lr_scheduler

class CustomModel(nn.Module):
  def __init__(self,checkpoint,num_labels): 
    super(CustomModel,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
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



tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")

data, train_data = load_data()
len_train_data = train_data['LABEL'].tolist()
tokenized_data = preprocess_data(data)
print(tokenized_data.keys)
#add label column to tokenized!!! 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)#return_tensors="tf"
train_dataloader,eval_dataloader = transform_data_asTR(tokenized_data, data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint="bert-base-german-cased",num_labels=2).to(device)
num_training_steps,num_epochs, metric, optimizer, lr_scheduler = set_params(len_train_data)
progress_bar_train = range(num_training_steps)
progress_bar_eval = range(num_epochs * len(eval_dataloader))


###training of model###
for epoch in range(num_epochs):
  model.train()
  for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      
      outputs = model(**batch)
    
      loss = outputs.loss
      loss.backward()

      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar_train.update(1)

### eval of model ###
#   model.eval()
#   for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])
#     progress_bar_eval.update(1)
    
#   print(metric.compute())


