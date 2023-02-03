# <Automated Labeler which gets trained on extracted topics of cleaned data..>
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

import pandas as pd
import os
import numpy as np
import pickle
import re
import operator
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import LabelingFunction
from sys import exit
from itertools import product
import random
import numpy as np
import logging 
import bayes_opt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def loggingdecorator(name):
            logger = logging.getLogger(name)
            def _decor(fn):
                function_name = fn.__name__
                def _fn(*args, **kwargs):
                    ret = fn(*args, **kwargs)
                    argstr = [str(x) for x in args]
                    argstr += [key+"="+str(val) for key,val in kwargs.items()]
                    logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
                    return ret
                return _fn
            return _decor


ABSTAIN = -1
AUTONOMOUS = 2
CONNECTIVITY = 3
DIGITALISATION = 4
ELECTRIFICATION = 5
INDIVIDUALISATION = 6
SHARED = 7
SUSTAINABILITY = 8

#DEFINE CLUSTERING MODEL (SIMPLE) K-Means###
#SELECT ALL KEYWORDS BASED ON SUFFICENT SOURCES INCLUDING LEVENSTHEIN DISTANCE###
#######################################################################################################################################################
def kMeans_cluster(x, label, kmeans, kmeans_vectorizer):    
    text = kmeans_vectorizer.transform([str(x)]).toarray()
    cluster = kmeans.predict(text)[0]+2

    if int(cluster) == int(label):
        return label
    else:
        return ABSTAIN

def make_cluster_lf(label, kmeans, kmeans_vectorizer):
    return LabelingFunction(
        name=f"cluster_{str(label)}",
        f=kMeans_cluster,
        resources=dict(label=label, kmeans =kmeans, kmeans_vectorizer=kmeans_vectorizer),
    )

def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    else:
        return ABSTAIN

def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{label}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

absolute_path = str(os.path.dirname(__file__)).split("src")[0] + r"files/Seed.xlsx"
seed_data = pd.read_excel(absolute_path,header = 0) 
df = seed_data[['AUTONOMOUS','ELECTRIFICATION','CONNECTIVITY','SHARED','SUSTAINABILITY','DIGITALISATION','INDIVIDUALISATION']]
autonomous_keywords = make_keyword_lf(keywords = df['AUTONOMOUS'].dropna().tolist(), label = AUTONOMOUS)
electrification_keywords = make_keyword_lf(keywords = df['ELECTRIFICATION'].dropna().tolist(), label = ELECTRIFICATION)
digitalisation_keywords = make_keyword_lf(keywords = df['DIGITALISATION'].dropna().tolist(), label = DIGITALISATION)
connectivity_keywords = make_keyword_lf(keywords = df['CONNECTIVITY'].dropna().tolist(), label = CONNECTIVITY)
sustainability_keywords = make_keyword_lf(keywords = df['SUSTAINABILITY'].dropna().tolist(), label = SUSTAINABILITY)
individualisation_keywords = make_keyword_lf(keywords = df['INDIVIDUALISATION'].dropna().tolist(), label = INDIVIDUALISATION)
shared_keywords = make_keyword_lf(keywords = df['SHARED'].dropna().tolist(), label = SHARED)

#load trained kMeans, fitted vectorizer for german kMeans
kmeans_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/kmeans_de.pkl", 'rb')) 
kmeans_vectorizer_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/kmeans_vectorizer_de.pkl", 'rb')) 
autonomous_cluster_d =  make_cluster_lf(label = AUTONOMOUS, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
electrification_cluster_d =  make_cluster_lf(label = ELECTRIFICATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
digitalisation_cluster_d =  make_cluster_lf(label = DIGITALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
connectivity_cluster_d =  make_cluster_lf(label = CONNECTIVITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
sustainability_cluster_d =  make_cluster_lf(label = SUSTAINABILITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
individualisation_cluster_d = make_cluster_lf(label = INDIVIDUALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
shared_cluster_d = make_cluster_lf(label = SHARED, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)

#load trained kMeans, fitted vectorizer for english kMeans
kmeans_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/kmeans_en.pkl", 'rb')) 
kmeans_vectorizer_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/kmeans_vectorizer_en.pkl", 'rb')) 
autonomous_cluster_e =  make_cluster_lf(label = AUTONOMOUS, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
electrification_cluster_e =  make_cluster_lf(label = ELECTRIFICATION, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
digitalisation_cluster_e =  make_cluster_lf(label = DIGITALISATION, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
connectivity_cluster_e =  make_cluster_lf(label = CONNECTIVITY, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
sustainability_cluster_e =  make_cluster_lf(label = SUSTAINABILITY, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
individualisation_cluster_e = make_cluster_lf(label = INDIVIDUALISATION, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)
shared_cluster_e = make_cluster_lf(label = SHARED, kmeans= kmeans_en, kmeans_vectorizer= kmeans_vectorizer_en)

class Labeler:
    """Class to label data automatically with help of Snorkel implementation. 
    """
    def __init__(self,lang:str,s_path:str, t_path:str, column:str):
        """Initialise a Label object that can label topics of cleaned texts.

        Args:
            lang (str): unicode of language to filter raw texts only in that language
            s_path (str): source path to file containing raw texts to clean
            t_path (str): target path to save file with cleaned texts
            column (str): selected text column to use for model training
        """
        # Create logger and assign handler
        logger = logging.getLogger("Labeler")

        handler  = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        filenames =  str(os.path.dirname(__file__)).split("src")[0] + r'doc\automated_labeling.log'
        fh = logging.FileHandler(filename=filenames)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(fh)

        logger = logging.getLogger("Labeler.randomSearch")
        logger.setLevel(logging.INFO)
        logger = logging.getLogger("Labeler.gridSearch")
        logger.setLevel(logging.INFO)
        logger = logging.getLogger("Labeler.bayesianOptim")
        logger.setLevel(logging.INFO)

        if lang == 'de':
            self.lfs = [
                autonomous_cluster_d,
                electrification_cluster_d,
                digitalisation_cluster_d,
                connectivity_cluster_d, 
                sustainability_cluster_d,
                individualisation_cluster_d, 
                shared_cluster_d,
                autonomous_keywords,
                electrification_keywords,
                digitalisation_keywords,
                connectivity_keywords,
                sustainability_keywords,
                individualisation_keywords,
                shared_keywords,
                ]
        if lang == 'en':
            self.lfs = [
                autonomous_cluster_e,
                electrification_cluster_e, 
                digitalisation_cluster_e, 
                connectivity_cluster_e,
                sustainability_cluster_e,
                individualisation_cluster_e,
                shared_cluster_e,
                autonomous_keywords,
                electrification_keywords,
                digitalisation_keywords,
                connectivity_keywords,
                sustainability_keywords,
                individualisation_keywords,
                shared_keywords,
                ]
         
        logger = logging.getLogger("Labeler")
        self.lang = lang
        self.update_data = False
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = column
        self.data = self.load_data()
        self.train_df, self.validate_df, self.test_df, self.train_test_df = self.generate_trainTestdata(lang)
        logger.info("Automated Labeling started with Language {l} and data file {path} (source) created. Target file is {tpath}".format(l = lang, path = s_path, tpath = t_path))
        logger.info(f"Automated Labeling applied on column: {self.text_col}")
    
    def load_data(self) -> pd.DataFrame:
        """Loads cleaned dataset containing topics as well.

        Returns:
            pd.DataFrame: Returns pandas DataFrame containing the cleaned texts and its topics
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path)
        return df.replace(np.nan, "",regex = False)

    def generate_trainTestdata(self, lang:str) -> pd.DataFrame:
        if lang == 'de':
            test_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_testset.xlsx"
            train_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_trainset.feather"
            val_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_valset.xlsx"
        elif lang == 'en':
            test_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_testset_en.xlsx"
            train_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_trainset_en.feather"
            val_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\label_valset_en.xlsx"

        if os.path.exists(test_path):
            test = pd.read_excel(test_path, index_col = 0)
            test = test[test['LABEL']!= 0]

            train = pd.read_feather(train_path)
            train = train.rename(columns = {'text':'TOPIC'})

            validate = pd.read_excel(val_path, index_col = 0)
            validate = validate[validate['LABEL']!= 0]
            
            test = test.replace(1,-1)
            validate = validate.replace(1,-1)
        else:
            self.data['LABEL'] = 0
            train, validate, test = np.split(self.data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(self.data)), int(.8*len(self.data))])
            train.reset_index(drop = True, inplace = True)
            test.reset_index(drop = True, inplace = True)
            validate.reset_index(drop = True, inplace = True)
            train.to_feather(train_path)
            test.to_excel(test_path)
            validate.to_excel(val_path)
            logger = logging.getLogger("Labeler")
            logger.info("Train, Test and Validate Dataset were generated. Please label train and validate data before further proceeding!")
            exit("No labeled Test and Validate data exist!")
        print(train.shape,test.shape,validate.shape)
        tr = train
        tr['LABEL'] = 0
        train_test = pd.concat([tr,test])
        train_test.reset_index(inplace = True)

        train = train.rename(columns = {self.text_col : 'text'})
        validate = validate.rename(columns = {self.text_col : 'text'})
        test = test.rename(columns = {self.text_col : 'text'})
        train_test = train_test.rename(columns = {self.text_col : 'text'})
        self.data = self.data.rename(columns = {self.text_col : 'text'})

        return train, validate, test, train_test

    def apply_labeling_functions(self):
        """First step of Snorkel. Previous defined Labeling Functions are applied to the loaded Training, Test and Validation Datasets.
        """
        logger = logging.getLogger("Labeler")
        logger.info("Application of Labeling Functions: {f}".format(f = self.lfs))
        self.applier = PandasLFApplier(lfs=self.lfs)

        #k_fold Cross Validation
        self.L_train_list =[]
        for j in range(2,7):
            training_folds = KFold(n_splits = j,shuffle = True, random_state = 12)
            i = 1
            for split in training_folds.split(self.train_test_df):
                fold_train_df = self.train_test_df.iloc[split[0]]
                
                fold_test_df = self.train_test_df.iloc[split[1]]
                fold_test_df = fold_test_df[fold_test_df['LABEL']!= 0]
                y_test = fold_test_df['LABEL'].to_numpy()

                l_train = self.applier.apply(df=fold_train_df)
                l_test = self.applier.apply(df=fold_test_df)
                self.L_train_list.append((l_train,l_test,y_test,j,i))

                # polarity = The set of unique labels this LF outputs (excluding abstains)
                # coverage = percentage of objects the LF labels
                # overlap  = percentage of objects with more than one label. 
                # conflict = percentage of objects with conflicting labels.
                L_analyis = LFAnalysis(L=l_train, lfs=self.lfs)

                logger.info(f"Training set coverage of {j}-fold Cross Validation with Trainingset {i}: {100 * L_analyis.label_coverage(): 0.1f}%")
            
                logger.info(L_analyis.lf_summary())

                i+=1
        
            # self.L_train = applier.apply(df=self.train_df)
        
        self.L_val = self.applier.apply(df=self.validate_df)
        self.L_test = self.applier.apply(df=self.test_df)


    def analysis_training_result(self):
        """Analysation of applied Labeling Functions and the coverage (in %) of each Function on the dataset.
        """
        logger = logging.getLogger("Labeler")
        logger.info('Validate metrics of trained Labeling Functions.')

        for L_train_fold in self.L_train_list:
            logger.info(f"Validate metrics of {L_train_fold[3]}-fold Cross Validation with Trainingset {L_train_fold[4]}")
            autonomous_cl, electrification_cl,digitalisation_cl,connectivity_cl, sustainability_cl,individualisation_cl, shared_cl,\
            autonomous_k,electrification_k,digitalisation_k,connectivity_k,sustainability_k,individualisation_k,shared_k = (L_train_fold[0] != ABSTAIN).mean(axis=0)

            logger.info(f"coverage_cluster_autonomous: {autonomous_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_electrification: {electrification_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_digitalisation: {digitalisation_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_connectivity: {connectivity_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_sustainability: {sustainability_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_individualisation: {individualisation_cl * 100:.1f}%")
            logger.info(f"coverage_cluster_shared: {shared_cl * 100:.1f}%")
            
            logger.info(f"coverage_keyword_autonomous: {autonomous_k * 100:.1f}%")
            logger.info(f"coverage_keyword_digitalisation: {digitalisation_k * 100:.1f}%")
            logger.info(f"coverage_keyword_electrification: {electrification_k * 100:.1f}%")
            logger.info(f"coverage_keyword_sustainability: {sustainability_k * 100:.1f}%")
            logger.info(f"coverage_keyword_connectivity: {connectivity_k * 100:.1f}%")
            logger.info(f"coverage_keyword_individualisation: {individualisation_k * 100:.1f}%")
            logger.info(f"coverage_keyword_shared: {shared_k * 100:.1f}%")


    def eval_labeling_model(self):
        """Evaluation of the best parameters for Snorkels Labeling Model by (Hyper-)parameter Tuning.
        Selected optimizer are: Grid Search, Random Search and Bayesian Optimization. 
        """
        
                
        self.evaluation_data = []
        for trainset in self.L_train_list:
            L_train_fold, L_test_fold,Y_test, k, i = trainset[0],trainset[1],trainset[2],trainset[3],trainset[4]
            logger = logging.getLogger("Labeler")
            logger.info(f"Evaluationg of best Model with {trainset[3]}-fold Cross Validation with Trainingset {trainset[4]}")

            #TODO: lr_scheduler: ["constant", "linear", "exponential", "step"]
            #random search
            loggerr = logging.getLogger("Labeler.randomSearch")
            try:
                self.apply_randomSearch(L_train_fold,L_test_fold,Y_test,k,i)
            except Exception as e:
                loggerr.warning("Error occurred: ", e)
            
            # #grid search
            loggerg = logging.getLogger("Labeler.gridSearch")
            try:
                self.apply_gridSearch(L_train_fold,L_test_fold,Y_test,k,i)
            except Exception as e:
                loggerg.warning("Error occurred: ", e)

            #bayesian search
            loggerb = logging.getLogger("Labeler.bayesianOptim")
            try:
                self.apply_bayesianOptimization(L_train_fold,L_test_fold,Y_test,k,i)
            except Exception as e:
                loggerb.warning("Error occurred: ", e)


        final_result_df = pd.DataFrame(self.evaluation_data).sort_values(by=['accuracy'], ascending=[False])
        final_param = final_result_df.to_dict('records')
        final_param = final_param[0]
        logger.info('\t'+ final_result_df.to_string().replace('\n', '\n\t'))
        
        logger.info(f"Best Model with {final_param['k-fold']}-fold Cross Validation identified: HPO: {final_param['Type']} Maximum reached accuracy: {final_param['accuracy']} with Parameters: {final_param}")

        
        test_set = [item[0] for item in self.L_train_list if (item[3] == final_param['k-fold'] and item[4] == final_param['trainingset'])]
        self.label_model = LabelModel(cardinality = 9, verbose=False)
        self.label_model.fit(L_train=test_set[0], n_epochs=int(final_param['n_epochs']), seed=123, log_freq=int(final_param['log_freq']), l2=final_param['l2'], lr=final_param['lr'], optimizer = final_param['optimizer'])

    @loggingdecorator("label.function")
    def apply_randomSearch(self,train,test,y_test,k,i, max_evals = 10):       
        logger = logging.getLogger("Labeler.randomSearch")
        #random.seed(123)
        #  Choose random hyperparameters until reach max evaluations
        results ={}
        for i in range(max_evals):
            p = {'n_epochs':np.random.randint(low = 10,high = 800),
            'log_freq':np.random.randint(low = 10,high = 200),
            'l2':round(np.random.uniform(low = 0.1,high = 2.0, size = 1)[0],1),
            'lr':round(np.random.uniform(low = 0.001,high = 0.02, size = 1)[0],3),
            'optimizer':np.random.choice(["sgd", "adam", "adamax"],1)[0]
            }
            #logger.info(f"Random choosen hyperparameters {p}")

            label_model = LabelModel(cardinality = 9, verbose=False)
            label_model.fit(L_train=train, n_epochs=p['n_epochs'], seed=123, log_freq=p['log_freq'], l2=p['l2'], lr=p['lr'], optimizer = p['optimizer'])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            results[(p['n_epochs'],p['log_freq'],p['l2'],p['lr'],p['optimizer'])]=label_model_acc
            self.evaluation_data.append({'Type':'RandomSearch','n_epochs':p['n_epochs'],'log_freq':p['log_freq'],'l2':p['l2'],'lr':p['lr'],'optimizer':p['optimizer'],'accuracy':label_model_acc,'k-fold':k,'trainingset':i})
        
        best_parameters = max(results.items(), key=operator.itemgetter(1))[0]
        highest_accuracy = results[best_parameters]

        logger.info(f"[Random Search]: Max. accuracy {highest_accuracy} with Parameters (n_epochs,log_freq,l2,lr): {best_parameters}")

    @loggingdecorator("label.function")
    def apply_gridSearch(self,train,test,y_test,k,i):
        logger = logging.getLogger("Labeler.gridSearch")
        
        hyperparameter_space ={
            'n_epochs':np.arange(50, 650, 100).tolist(),
            'log_freq':np.arange(10, 220, 50).tolist(),
            'l2':np.arange(0.1, 0.6, 0.1).tolist(),
            'lr':np.arange(0.002, 0.012, 0.002).tolist(), 
            'optimizer':["sgd", "adam", "adamax"]
        }
        permutations = list(product(hyperparameter_space['n_epochs'],hyperparameter_space['log_freq'],hyperparameter_space['l2'],hyperparameter_space['lr'],hyperparameter_space['optimizer']))
        # permutations = {k: v for k, v in hyperparameter_space.items()}
        results = {}
        
        for p in permutations:
            #logger.info(f"Selected parameter {p}")
            label_model = LabelModel(cardinality = 9, verbose=False)
            label_model.fit(L_train=train, n_epochs=p[0], seed=123, log_freq=p[1], l2=p[2], lr=p[3], optimizer = p[4])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            results[(p[0],p[1],p[2],p[3], p[4])]=label_model_acc
            self.evaluation_data.append({'Type':'GridSearch','n_epochs':p[0],'log_freq':p[1],'l2':p[2],'lr':p[3],'optimizer':p[4],'accuracy':label_model_acc,'k-fold':k,'trainingset':i})

        best_parameters = max(results.items(), key=operator.itemgetter(1))[0]
        highest_accuracy = results[best_parameters]

        logger.info(f"[Grid Search]: Max. accuracy {highest_accuracy} with Parameters (n_epochs,log_freq,l2,lr): {best_parameters}")

    @loggingdecorator("label.function")
    def apply_bayesianOptimization(self,train,test,y_test,k,i):
        logger = logging.getLogger("Labeler.bayesianOptim")
        label_model = LabelModel(cardinality = 9, verbose=False)
        hyperparameter_space ={
            'n_epochs':[10,800],
            'log_freq':[10,200],
            'l2':[0.1,2.0],
            'lr':[0.001,0.2],
            'optimizer':[0,2]
        }

        def model_train(n_epochs,log_freq,l2,lr, optimizer):
            if round(optimizer) == 0:
                optim = "sgd"
            elif round(optimizer) == 1:
                optim = "adam"
            elif round(optimizer) == 2:
                optim = "adamax"
            else:
                optim = "adam"
            label_model.fit(L_train=train, n_epochs=round(n_epochs), seed=123, log_freq=round(log_freq), l2=l2, lr=lr, optimizer = optim)
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            self.evaluation_data.append({'Type':'BayesianOptim','n_epochs':round(n_epochs),'log_freq':round(log_freq),'l2':l2,'lr':lr,'optimizer':optim,'accuracy':label_model_acc,'k-fold':k,'trainingset':i})
            return label_model_acc

        optimizer = bayes_opt.BayesianOptimization(f=model_train,pbounds =hyperparameter_space ,verbose = 1, random_state = 4)
        optimizer.maximize(init_points = 5, n_iter = 50)        

        best_parameters = optimizer.max["params"]
        highest_accuracy = optimizer.max["target"]

        logger.info(f"[Bayesian Optimization]: Max. accuracy {highest_accuracy} with Parameters: {best_parameters}")

    def test_model(self,model):
        preds_test_label = model.predict(L=self.L_test)
        test_df = self.test_df
        test_df['LABEL'] = preds_test_label
        # before_test_shape = test_df.shape
        test_df = test_df[test_df['LABEL'] != -1]
        after_test_shape = test_df.shape
        return after_test_shape
        
    def validate_model(self):
        logger = logging.getLogger("Labeler")
        preds_val_label = self.label_model.predict(L=self.L_val)
        Y_val = self.validate_df['LABEL'].to_numpy()
        validate_acc = self.label_model.score(L=self.L_val, Y=Y_val, tie_break_policy="random")["accuracy" ]
        logger.info(f"Accuracy on Validation set: {validate_acc}")
        
        validation_df = self.validate_df
        validation_df['LABEL'] = preds_val_label
        before_val_shape = self.validate_df.shape
        validation_df = validation_df[validation_df['LABEL'] != -1]
        after_val_shape = validation_df.shape
        print(f"Shape of Validationset before: {before_val_shape} and after: {after_val_shape}")
       

    def apply_trained_model(self):
        logger = logging.getLogger("Labeler")
        # logger.info(f"Langauge:{self.lang}. Applying trained model with best parameters and trainset on test set.")
        self.validate_model()

        self.L_data = self.applier.apply(df= self.data)
        logger.info(f"Langauge:{self.lang}. Applying trained model with best parameters and trainset on whole data set.")
        label = self.label_model.predict(L=self.L_data)
        self.data['LABEL'] = label
        before_data_shape = self.data.shape
        self.data = self.data[self.data['LABEL'] != -1]
        after_data_shape = self.data.shape
        logger.info(f"Shape of Dataset before: {before_data_shape} and after: {after_data_shape}")

    def save_data(self):
        """Saves labeled data as feather into files folder.
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        self.data.reset_index(inplace=True)
        if os.path.exists(path):            
            if self.update_data:
                self.data.to_feather(path)
        self.data.to_feather(path)  

    def save_model(self):
        """Saving of trained Label model as pickle file with optimized parameter.
        """
        logger = logging.getLogger("Labeler")
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(self.lang)+r".pkl"

        Y_val = self.validate_df['LABEL'].to_numpy() 
        new_model_acc = self.label_model.score(L=self.L_val, Y=Y_val, tie_break_policy="random")["accuracy" ]
        
        if os.path.exists(path):            
            old_label_model = LabelModel(cardinality = 9, verbose=False)
            # L_train_dummy = np.random.randint(-1, 2, size=(10**6, 10), dtype=np.int8)
            
            old_label_model.load(path)
            old_model_acc = old_label_model.score(L=self.L_val, Y=Y_val, tie_break_policy="random")["accuracy" ] 

            if new_model_acc > old_model_acc:
                self.label_model.save(path)
                self.update_data = True

                logger.info(f"New model accuracy: {new_model_acc}. Saved model accuracy: {old_model_acc} (Validation Set)")
                logger.info("Model with better accuracy saved!")
        else:
            self.label_model.save(path)
            logger.info(f"Model accuracy: {new_model_acc}. (Validation Set)")
            logger.info("Model saved!")
  
    def show_samples_per_class(self):
        df1 = self.data.iloc[self.L_train[:, 1] == ELECTRIFICATION].sample(10, random_state=1)[['text','TOPIC']]
        #df2 = self.data.iloc[self.L_train[:, 1] == AUTONOMOUS].sample(10, random_state=1)[['text','TOPIC']]
        # df3 = self.data.iloc[self.L_train[:, 1] == DIGITALISATION].sample(10, random_state=1)[['text','TOPIC']]
        #df4 = self.data.iloc[self.L_train[:, 1] == INDIVIDUALISATION].sample(10, random_state=1)[['text','TOPIC']]

        
    def assign_labels_final(self):
        list_of_values = {"ABSTAIN":ABSTAIN,"AUTONOMOUS":AUTONOMOUS,"ELECTRIFICATION":ELECTRIFICATION,"CONNECTIVITY":CONNECTIVITY,"SHARED":SHARED,\
            "SUSTAINABILITY":SUSTAINABILITY,"DIGITALISATION":DIGITALISATION,"INDIVIDUALISATION":INDIVIDUALISATION}
        for v in list_of_values.keys:
            self.data.loc[(self.data['LABEL']==list_of_values[v]), 'LABEL'] = v
    
    def run(self):
        self.apply_labeling_functions()
        self.analysis_training_result()
        self.eval_labeling_model()
        self.apply_trained_model()
        #self.assign_labels_final()
        #self.show_samples_per_class()
        self.save_model()
        self.save_data()

if __name__ == "__main__":
    l = Labeler('de',r"files\topiced_texts.feather",r"files\labeled_texts.feather",'URL_TEXT')
    l.run()
    l = Labeler('de',r"files\topiced_texts.feather",r"files\labeled_texts.feather",'TOPIC')
    l.run()
    e = Labeler('en',r"files\topiced_texts_en.feather",r"files\labeled_texts_en.feather",'URL_TEXT')
    e.run()
    e = Labeler('en',r"files\topiced_texts_en.feather",r"files\labeled_texts_en.feather",'TOPIC')
    e.run()