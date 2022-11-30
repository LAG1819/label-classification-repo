import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import numpy as np

tf.get_logger().setLevel('ERROR')

class Transformer():
    def __init__(self, bert_model_name = 'bert_multi_cased_L-12_H-768_A-12'):
        self.data = None
        self.bert_model_name = bert_model_name
        self.classifier_model = None
        self.history = None
        self.train_df, self.test_df, self.val_df = self.load_data()
        self.tfhub_handle_encoder, self.tfhub_handle_preprocess = self.select_bert_model()
        self.optimizer, self.loss, self.metrics, self.epochs = self.create_set_up()

    def load_data(self):
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32

        df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\Output_texts_labeled.csv")
        df = pd.read_csv(df_path, header = 0, delimiter=",")
        self.data = df.replace(np.nan, "",regex = False)
        self.data['text'] = self.data['text'].apply(lambda row: row.replace("|","."))
        self.data['sentences'] = self.data['text']
        
        target = self.data.pop('LABEL')
        features = self.data[['sentences']]

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
       
        features = ['sentences']

        train_ds = (tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(train_ds[features].values, tf.string),
                    tf.cast(train_ds['LABEL'].values, tf.float32)
                )
            )
        ).batch(batch_size=batch_size)
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = (tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(test_ds[features].values, tf.string),
                    tf.cast(test_ds['LABEL'].values, tf.float32)
                )
            )
        ).batch(batch_size=batch_size)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = (tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(val_ds[features].values, tf.string),
                    tf.cast(val_ds['LABEL'].values, tf.float32)
                )
            )
        ).batch(batch_size=batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds,test_ds,val_ds
    
    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target').astype('float64')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    def load_data_keras(self):
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32
        seed = 42
        df_path = os.path.join(str(os.path.dirname(__file__)).split("src")[0],"files\Output_texts_labeled.csv")

        ds = tf.keras.utils.get_file(df_path).shuffle(buffer_size=batch_size)
        print(ds)

        raw_train_ds = tf.keras.utils.get_file(
            df_path)
            #batch_size=batch_size,
            #validation_split=0.2,
            #subset='training',
            #seed=seed)

        class_names = raw_train_ds.class_names
        train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        val_ds = tf.keras.utils.text_dataset_from_directory(
            df_path,
            batch_size=batch_size,
            validation_split=0.2,
            subset='validation',
            seed=seed)

        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds = tf.keras.utils.text_dataset_from_directory(
            df_path,
            batch_size=batch_size)

        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, test_ds, val_ds

    def save_data(self):
        self.data.to_csv(str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_classified.csv", index = False)

    def create_set_up(self):
        epochs = 3
        steps_per_epoch = tf.data.experimental.cardinality(self.train_df).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.metrics.CategoricalAccuracy()

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')
        return optimizer, loss, metrics, epochs


    def select_bert_model(self):
        map_name_to_handle = {
            'bert_en_uncased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'bert_en_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
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

    def model_preprocess_test(self):
        bert_model = hub.KerasLayer(self.tfhub_handle_encoder)
        bert_preprocess_model = hub.KerasLayer(self.tfhub_handle_preprocess)

        text_test = self.data['text'].tolist()[0]
        text_test = text_test.replace("|",".")
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

        classifier_model = c.create_classifier_model()
        bert_raw_result = classifier_model(tf.constant(text_test))
        print(tf.sigmoid(bert_raw_result))        

    def create_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentences')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(5, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)  

    def apply_classifier(self):
        self.classifier_model = self.create_classifier_model()
        self.classifier_model.compile(optimizer= self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics)
        print(f'Training model with {self.tfhub_handle_encoder}')

        self.history = self.classifier_model.fit(x=self.train_df,
                                    validation_data=self.val_df,
                                    epochs=self.epochs, 
                                    batch_size = 1)

    def save_model(self):
        dataset_name = 'imdb'
        saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

        self.classifier_model.save(saved_model_path, include_optimizer=False)

    def load_model(self):
        dataset_name = 'imdb'
        saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
        reloaded_model = tf.saved_model.load(saved_model_path)
        return reloaded_model
    
    def evaluate_model(self):
        loss, accuracy = self.classifier_model.evaluate(self.test_df)

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
        c.apply_classifier()

if __name__ == "__main__":
    c = Transformer()
    # for text_batch, label_batch in c.train_df.take(1):
    #     for i in range(1):
    #         print(f'Review: {text_batch.numpy()[i]}')
    
    # c.model_preprocess_test()
    c.apply_classifier()
    c.evaluate_model()

    
    