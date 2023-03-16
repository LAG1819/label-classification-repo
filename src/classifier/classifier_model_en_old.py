# <BERT-based text classification in Keras. Classification modell is trained on topics in englisch. Deprecated due to performance issues but working layout.>
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

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
# from bohb import BOHB
# import bohb.configspace as cs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import numpy as np
import logging
from sklearn.model_selection import KFold 
import bayes_opt
import keras_tuner as kt

tf.get_logger().setLevel('ERROR')

class Transformer():
    def __init__(self, bert_model_name = 'bert_multi_cased_L-12_H-768_A-12',text_col ='URL_TEXT',lang = 'en'):
        # Create logger and assign handler
        logger = logging.getLogger("Labeler")

        handler  = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        self.text_col = text_col
        self.lang = lang
        self.bert_model_name = bert_model_name
        self.classifier_model = None
        self.history = None
        self.batch_size = 32
        self.AUTOTUNE = tf.data.AUTOTUNE

        self.tfhub_handle_encoder, self.tfhub_handle_preprocess = self.select_bert_model()
        self.data,self.val_df, self.val_label = self.load_data()
       
    def select_bert_model(self):
        map_name_to_handle = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_base/2',
            'electra_small':
                'https://tfhub.dev/google/electra_small/2',
            'electra_base':
                'https://tfhub.dev/google/electra_base/2',
            'experts_pubmed':
                'https://tfhub.dev/google/experts/bert/pubmed/2',
            'experts_wiki_books':
                'https://tfhub.dev/google/experts/bert/wiki_books/2',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        }
        map_model_to_preprocess = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-2_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-4_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-6_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-8_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-10_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-128_A-2':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-256_A-4':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-512_A-8':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'small_bert/bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'albert_en_base':
                'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
            'electra_small':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'electra_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_pubmed':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'experts_wiki_books':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
            'talking-heads_base':
                'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        }

        tfhub_handle_encoder = map_name_to_handle[self.bert_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[self.bert_model_name]

        print(f'BERT model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
        return tfhub_handle_encoder, tfhub_handle_preprocess

    def load_data(self):
        # df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\label_testset_en.xlsx")
        # df = pd.read_excel(df_path, index_col = 0)

        df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\labeled_texts_en.feather")
        df = pd.read_feather(df_path)
        data = df.replace(np.nan, "",regex = False)
        data = data[:1000]
        data['sentences'] = data[self.text_col]
        
        train, validate, test = np.split(data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(data)), int(.8*len(data))])
        data = pd.concat([train,test])

        features = ['sentences']

        val_ds = (tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(validate[features].values, tf.string),
                    tf.cast(validate['LABEL'].values, tf.float32)
                )
            )
        ).batch(batch_size=self.batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

        val_label = (tf.data.Dataset.from_tensor_slices(
                (
                    # tf.cast(validate[features].values, tf.string),
                    tf.cast(validate['LABEL'].values, tf.float32)
                )
            )
        ).batch(batch_size=self.batch_size)
        val_label = val_label.cache().prefetch(buffer_size=self.AUTOTUNE)
    
        return data, val_ds,val_label

    def model_preprocess_test(self):
        bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess)
        bert_model = hub.KerasLayer(self.tfhub_handle_encoder,trainable=True)

        text_test = self.data[self.text_col].tolist()[0]
        text_test = [text_test]
        print("Test: "+str(text_test))

        text_preprocessed = bert_preprocess_model(text_test)

        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

        bert_results = bert_model(text_preprocessed)

        print(f'Loaded BERT: {self.tfhub_handle_encoder}')
        print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
        print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
        print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
        print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

        classifier_model = self.create_classifier_model()
        bert_raw_result = classifier_model(tf.constant(text_test))
        print(tf.sigmoid(bert_raw_result))

    def split_data(self,k=2):
        train_test_list =[]

        #target = self.data.pop('LABEL')
        features = ['sentences']
         # features = data[['sentences','TOPICS']]

        training_folds = KFold(n_splits = k,shuffle = True, random_state = 12)
        for i,split in enumerate(training_folds.split(self.data)):
            train_ds = self.data.iloc[split[0]]
            test_ds = self.data.iloc[split[1]]

        # # # set aside 20% of train and test data for evaluation
        # train_ds, test_ds, train_label, test_label = train_test_split(features, target,
        #     test_size=0.2, shuffle = True, random_state = 8)

        # # # Use the same function above for the validation set
        # train_ds, val_ds, train_label, val_label = train_test_split(train_ds, train_label, 
        #     test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2
        
        # print(train_ds.shape,val_ds.shape,test_ds.shape)

        # train_ds['LABEL'] = train_label.to_frame()['LABEL'].tolist()
        # val_ds['LABEL'] = val_label.to_frame()['LABEL'].tolist()
        # test_ds['LABEL'] = test_label.to_frame()['LABEL'].tolist()

            train_label = (tf.data.Dataset.from_tensor_slices(
                    (
                        # tf.cast(train_ds[features].values, tf.string),
                        tf.cast(train_ds['LABEL'].values, tf.float32)
                    )
                )
            ).batch(batch_size=self.batch_size)
            train_label = train_label.cache().prefetch(buffer_size=self.AUTOTUNE)
            
            train_ds = (tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(train_ds[features].values, tf.string),
                        tf.cast(train_ds['LABEL'].values, tf.float32)
                    )
                )
            ).batch(batch_size=self.batch_size)
            train_ds = train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

            test_label = (tf.data.Dataset.from_tensor_slices(
                    (
                        # tf.cast(test_ds[features].values, tf.string),
                        tf.cast(test_ds['LABEL'].values, tf.float32)
                    )
                )
            ).batch(batch_size=self.batch_size)
            test_label = test_label.cache().prefetch(buffer_size=self.AUTOTUNE)

            test_ds = (tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(test_ds[features].values, tf.string),
                        tf.cast(test_ds['LABEL'].values, tf.float32)
                    )
                )
            ).batch(batch_size=self.batch_size)
            test_ds = test_ds.cache().prefetch(buffer_size=self.AUTOTUNE)            

            train_test_list.append([train_ds,train_label,test_ds, test_label, k,i])
        return train_test_list
    
    # # A utility method to create a tf.data dataset from a Pandas Dataframe
    # def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    #     dataframe = dataframe.copy()
    #     labels = dataframe.pop('target').astype('float64')
    #     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    #     if shuffle:
    #         ds = ds.shuffle(buffer_size=len(dataframe))
    #     ds = ds.batch(batch_size)
    #     return ds

    # def load_data_keras(self):
        # AUTOTUNE = tf.data.AUTOTUNE
        # batch_size = 32
        # seed = 42
        # df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\Output_texts_labeled.csv")

        # ds = tf.keras.utils.get_file(df_path).shuffle(buffer_size=batch_size)
        # print(ds)

        # raw_train_ds = tf.keras.utils.get_file(
        #     df_path)
        #     #batch_size=batch_size,
        #     #validation_split=0.2,
        #     #subset='training',
        #     #seed=seed)

        # class_names = raw_train_ds.class_names
        # train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # val_ds = tf.keras.utils.text_dataset_from_directory(
        #     df_path,
        #     batch_size=batch_size,
        #     validation_split=0.2,
        #     subset='validation',
        #     seed=seed)

        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # test_ds = tf.keras.utils.text_dataset_from_directory(
        #     df_path,
        #     batch_size=batch_size)

        # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # return train_ds, test_ds, val_ds





    def create_set_up(self, train_df,lr = 3e-5,epoch = 3,optimizer_t = 'adamw'):
        init_lr = lr 
        epochs = epoch
        steps_per_epoch = tf.data.experimental.cardinality(train_df).numpy()

        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.metrics.CategoricalAccuracy()

        
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type=optimizer_t)
        return optimizer, loss, metrics, epochs        

    def create_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(9, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)  

    def apply_classifier(self, train_df, test_df,optimizer, loss, metrics, epochs):
        classifier_model = self.create_classifier_model()
        classifier_model.compile(optimizer= optimizer,
                            loss=loss,
                            metrics=metrics)
        print(f'Training model with {self.tfhub_handle_encoder}')

        classifier_model.fit(train_df, validation_data = test_df,                            
                            epochs=epochs, 
                            batch_size = 1,
                            validation_split=0.2)
        return classifier_model

    def evaluate_classifier(self):
        for k in range (2,3):
            #train_test_set = train_ds,train_label,test_ds, test_label, k,i
            for i,train_test_set in enumerate(self.split_data(k)):
                print(f"{k}-fold Split with train-test set {i}")
                
                train_df = train_test_set[0]
                train_label = train_test_set[1]
                test_df = train_test_set[2]
                test_label = train_test_set[3]
                
                self.apply_bayesian(train_df,test_df)
                self.apply_random_search(train_df,test_df)
                self.apply_hyperband(train_df,test_df)
                break

            # optimizer, loss, metrics, epochs = self.create_set_up(train_df)
            # classifier_model = self.apply_classifier(train_df,validation_data = test_df,optimizer, loss, metrics, epochs)
            # loss, accuracy = classifier_model.evaluate(test_df)

    def apply_hyperband(self, train_df, test_df):
        def model_builder(hp):
            model = self.create_classifier_model()

            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            optimizer, loss, metrics, epochs = self.create_set_up(train_df,lr = hp_learning_rate,epoch = 3,optimizer_t = 'adamw')

            model.compile(optimizer= optimizer,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=[tf.metrics.CategoricalAccuracy()])
            return model
        
        print(model_builder(kt.HyperParameters()))
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\model_tuning_hyperband_en_"   
        dir = path + r"0"
        while os.path.exists(dir):
            number = int(dir.split("_")[-1]) + 1
            dir = path+str(number)
        
        tuner_hyperband = kt.Hyperband(model_builder,
                     objective= kt.Objective("val_categorical_accuracy", direction="max"),
                     max_epochs=1,
                     factor=3,
                     directory=dir,
                     project_name='hyperband_tuner'+str(number))

        
        tuner_hyperband.search(x = train_df, validation_data = test_df,                            
                            epochs=3,
                            shuffle = True, 
                            batch_size = 1,
                            validation_split=0.2, 
                            callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner_hyperband.get_best_hyperparameters(num_trials=1)[0]

        model = tuner_hyperband.hypermodel.build(best_hps)
        history = model.fit(train_df, validation_data = test_df, epochs=1)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        print(f"""
        Hyperband Tuning completed. 
        Optimal learning rate:{best_hps.get('learning_rate')}
        Optimal epoch: {best_epoch}.
        """)
        # logger.info(f"[Bayesian Optimization]: Max. accuracy {highest_accuracy} with Parameters: {best_parameters}")

    def apply_random_search(self, train_df, test_df):
        def model_builder(hp):
            model = self.create_classifier_model()

            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            optimizer, loss, metrics, epochs = self.create_set_up(train_df,lr = hp_learning_rate,epoch = 3,optimizer_t = 'adamw')

            model.compile(optimizer= optimizer,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=[tf.metrics.CategoricalAccuracy()])
            return model
        
        print(model_builder(kt.HyperParameters()))
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\model_tuning_randomS_en_"
        dir = path + r"0"
        number = 0
        while os.path.exists(dir):
            number = int(dir.split("_")[-1]) + 1
            dir = path+str(number)

        tuner_random_search = kt.RandomSearch(model_builder,
                     objective= kt.Objective("categorical_accuracy", direction="max"),#val_categorical_accuracy
                     max_trials=4,
                     seed=3,
                     directory=dir,
                     project_name='randomSearch_tuner'+str(number))

        
        tuner_random_search.search(x = train_df, validation_data = test_df,                            
                            epochs=3,
                            shuffle = True, 
                            batch_size = 1,
                            validation_split=0.2, 
                            callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner_random_search.get_best_hyperparameters(num_trials=1)[0]

        model = tuner_random_search.hypermodel.build(best_hps)
        history = model.fit(train_df, validation_data = test_df, epochs=1)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        print(f"""
        Random Search Tuning completed. 
        Optimal learning rate:{best_hps.get('learning_rate')}
        Optimal epoch: {best_epoch}.
        """)
        # logger.info(f"[Bayesian Optimization]: Max. accuracy {highest_accuracy} with Parameters: {best_parameters}")

    def apply_bayesian(self, train_df, test_df):
        def model_builder(hp):
            model = self.create_classifier_model()

            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            optimizer, loss, metrics, epochs = self.create_set_up(train_df,lr = hp_learning_rate,epoch = 3,optimizer_t = 'adamw')

            model.compile(optimizer= optimizer,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=[tf.metrics.CategoricalAccuracy()])
            return model
        
        print(model_builder(kt.HyperParameters()))
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classification\model_tuning_bayesian_en_"   
        dir = path + r"0"
        number = 0
        while os.path.exists(dir):
            number = int(dir.split("_")[-1]) + 1
            dir = path+str(number)
        
        tuner_bayesian = kt.BayesianOptimization(model_builder,
                     objective= kt.Objective("val_categorical_accuracy", direction="max"),
                     max_trials=2,
                     seed=3,
                     directory=dir,
                     project_name='bayesian_tuner'+str(number))

        
        tuner_bayesian.search(x = train_df, validation_data = test_df,                            
                            epochs=3,
                            shuffle = True, 
                            batch_size = 1,
                            validation_split=0.2, 
                            callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner_bayesian.get_best_hyperparameters(num_trials=1)[0]

        model = tuner_bayesian.hypermodel.build(best_hps)
        history = model.fit(train_df, validation_data = test_df, epochs=1)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        print(f"""
        Bayesian Tuning completed. 
        Optimal learning rate:{best_hps.get('learning_rate')}
        Optimal epoch: {best_epoch}.
        """)
        # logger.info(f"[Bayesian Optimization]: Max. accuracy {highest_accuracy} with Parameters: {best_parameters}")    
   

    def save_model(self):
        saved_model_path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classifier\bert_model_"+str(self.lang)
        # dataset_name = 'imdb'
        # saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
        

        self.classifier_model.save(saved_model_path, include_optimizer=False)

    def load_model(self):
        saved_model_path = str(os.path.dirname(__file__)).split("src")[0] + r"models\classifier\bert_model_"+str(self.lang)
        # dataset_name = 'imdb'
        # saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
        reloaded_model = tf.saved_model.load(saved_model_path)
        return reloaded_model
    
    def evaluate_model_plot(self, classifier_model):
        loss, accuracy = classifier_model.evaluate(self.test_df)

        history_dict = self.history.history
        print(history_dict.keys())

        acc = history_dict['categorical_accuracy']
        val_acc = history_dict['val_categorical_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def run(self):
        # self.model_preprocess_test()
        self.evaluate_classifier()
        self.apply_classifier()

if __name__ == "__main__":
    c = Transformer()
    c.run()
   
    
    # for text_batch in c.val_df.take(1):
    #     for i in range(1):
    #         print(text_batch[i])
    # for label_batch in c.val_label.take(1):
    #     for i in range(1):
    #         print(label_batch[i])
        
    

    
    