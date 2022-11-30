from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
import pandas as pd
import numpy as np
from transformers import create_optimizer
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")

def load_data():
    df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\Output_texts_labeled.csv")
    df = pd.read_csv(df_path, header = 0, delimiter=",")
    data = df.replace(np.nan, "",regex = False)
    data['text'] = data['text'].apply(lambda row: row.replace("|","."))
    return data

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def transform_data_asTF(tokenized_data):
    
    target = tokenized_data.pop('LABEL')
    features = tokenized_data[['sentences']]

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

    train = train_ds[['LABEL','text']].to_dict('records')
    test = test_ds[['LABEL','text']].to_dict('records')

    tf_train_set = model.prepare_tf_dataset(
    train,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        test,
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    return tf_train_set,tf_validation_set, train


data = load_data()
tokenized_data = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_train_set,tf_validation_set, train_ds = load_data_asTF(tokenized_data)

batch_size = 16
num_epochs = 5
batches_per_epoch = len(train_ds) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)

