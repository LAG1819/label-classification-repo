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


ABSTAIN = 1
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
kmeans_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_de.pkl", 'rb')) 
kmeans_vectorizer_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_vectorizer_de.pkl", 'rb')) 
autonomous_cluster_d =  make_cluster_lf(label = AUTONOMOUS, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
electrification_cluster_d =  make_cluster_lf(label = ELECTRIFICATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
digitalisation_cluster_d =  make_cluster_lf(label = DIGITALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
connectivity_cluster_d =  make_cluster_lf(label = CONNECTIVITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
sustainability_cluster_d =  make_cluster_lf(label = SUSTAINABILITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
individualisation_cluster_d = make_cluster_lf(label = INDIVIDUALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
shared_cluster_d = make_cluster_lf(label = SHARED, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)

#load trained kMeans, fitted vectorizer for english kMeans
kmeans_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_en.pkl", 'rb')) 
kmeans_vectorizer_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_vectorizer_en.pkl", 'rb')) 
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
    def __init__(self,lang:str,s_path:str, t_path:str, k = 3):
        """Initialise a Label object that can label topics of cleaned texts.

        Args:
            lang (str): unicode of language to filter raw texts only in that language
            s_path (str): source path to file containing raw texts to clean
            t_path (str): target path to save file with cleaned texts
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
            # self.lfs = [autonomous_keywords,electrification_keywords,digitalisation_keywords]
            self.lfs = [autonomous_cluster_d, electrification_cluster_d,digitalisation_cluster_d,connectivity_cluster_d, sustainability_cluster_d,individualisation_cluster_d, shared_cluster_d,\
                autonomous_keywords,electrification_keywords,digitalisation_keywords,connectivity_keywords,sustainability_keywords,individualisation_keywords,shared_keywords]
        if lang == 'en':
            self.lfs = [autonomous_cluster_e, electrification_cluster_e, digitalisation_cluster_e, connectivity_cluster_e,sustainability_cluster_e,individualisation_cluster_e,shared_cluster_e,\
                autonomous_keywords,electrification_keywords,digitalisation_keywords,connectivity_keywords,sustainability_keywords,individualisation_keywords,shared_keywords]
         
        logger = logging.getLogger("Labeler")

        self.k = k
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = 'TOPIC'#'URL_TEXT'
        self.data = self.load_data()
        self.train_df, self.validate_df, self.test_df = self.generate_trainTestdata(lang)
        logger.info("Automated Labeling started with Language {l} and data file {path} (source) created. Target file is {tpath}".format(l = lang, path = s_path, tpath = t_path))
    
    def load_data(self) -> pd.DataFrame:
        """Loads cleaned dataset containing topics as well.

        Returns:
            pd.DataFrame: Returns pandas DataFrame containing the cleaned texts and its topics
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path)
        df = df.rename(columns = {self.text_col : 'text'})
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
            validate = pd.read_excel(val_path, index_col = 0)
            validate = validate[validate['LABEL']!= 0]
        else:
            self.data['LABEL'] = 0
            train, validate, test = np.split(self.data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(self.data)), int(.8*len(self.data))])
            train.reset_index(drop = True, inplace = True)
            test.reset_index(drop = True, inplace = True)
            validate.reset_index(drop = True, inplace = True)
            train.to_feather(train_path)
            test.to_excel(test_path)
            validate.to_excel(val_path)
            print("Train, Test and Validate Dataset were generated. Please label train and validate data before further proceeding!")
            exit("No labeled Test and Validate data exist!")
        print(train.shape,test.shape,validate.shape)
        return train, validate, test

    def save_data(self):
        """Saves labeled data as feather into files folder.
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

    def apply_labeling_functions(self):
        """First step of Snorkel. Previous defined Labeling Functions are applied to the loaded Training, Test and Validation Datasets.
        """
        logger = logging.getLogger("Labeler")
        logger.info("Application of Labeling Functions: {f}".format(f = self.lfs))

        #k_fold Cross Validation
        applier = PandasLFApplier(lfs=self.lfs)
        training_folds = KFold(n_splits = self.k,shuffle = True, random_state = 12)
        self.L_train_list =[]
        i = 1
        for split in training_folds.split(self.train_df):
            fold_train_df = self.train_df.iloc[split[0]]
            l_train = applier.apply(df=fold_train_df)
            self.L_train_list.append(l_train)

            L_analyis = LFAnalysis(L=l_train, lfs=self.lfs)

            logger.info(f"Training set coverage of {self.k}-fold Cross Validation with Trainingset {i}: {100 * L_analyis.label_coverage(): 0.1f}%")
            logger.info(L_analyis.lf_summary())
            i+=1
        # self.L_train = applier.apply(df=self.train_df)
        #self.L_val = applier.apply(df=self.validate_df)
        self.L_test = applier.apply(df=self.test_df)


    def analysis_training_result(self):
        """Analysation of applied Labeling Functions and the coverage (in %) of each Function on the dataset.
        """
        logger = logging.getLogger("Labeler")
        logger.info('Validate metrics of trained Labeling Functions.')
        i = 0
        for L_train_fold in self.L_train_list:
            logger.info(f"Validate metrics of {self.k}-fold Cross Validation with Trainingset {i}")
            autonomous_cl, electrification_cl,digitalisation_cl,connectivity_cl, sustainability_cl,individualisation_cl, shared_cl,\
                    autonomous_k,electrification_k,digitalisation_k,connectivity_k,sustainability_k,individualisation_k,shared_k = (L_train_fold != ABSTAIN).mean(axis=0)

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
            
        #print(self.label_model.score(L_validate, Y=validate_labels,metrics=["f1","accuracy",'precision','recall']))

    def eval_labeling_model(self):
        """Evaluation of the best parameters for Snorkels Labeling Model by (Hyper-)parameter Tuning.
        Selected optimizer are: Grid Search, Random Search and Bayesian Optimization. 
        """
        Y_test = self.test_df['LABEL'].to_numpy()

        best_model_list = {}
        best_param_list = {}
        self.evaluation_data = []
        i=0
        for L_train_fold in self.L_train_list:
            logger = logging.getLogger("Labeler")
            logger.info(f"Evaluationg of best Model with {self.k}-fold Cross Validation with Trainingset {i}")
            #random search
            loggerr = logging.getLogger("Labeler.randomSearch")
            highest_acc = None
            best_param  = None
            try:
                rand_param, rand_acc = self.apply_randomSearch(L_train_fold , self.L_test,Y_test)
                loggerr.info(f"Max. accuracy {rand_acc} with Parameters (n_epochs,log_freq,l2,lr):{rand_param}")
            except Exception as e:
                loggerr.info("Error occurred: ", e)
                rand_param, rand_acc = {'n_epochs':1,'log_freq':1,'l2':1,'lr':1},0

            highest_acc = rand_acc
            best_param = {'n_epochs':rand_param[0],'log_freq':rand_param[1],'l2':rand_param[2],'lr':rand_param[3]}
            #grid search
            loggerg = logging.getLogger("Labeler.gridSearch")
            try:
                grid_param, grid_acc = self.apply_gridSearch(L_train_fold , self.L_test,Y_test)
                loggerg.info(f"Max. accuracy {grid_acc} with Parameters (n_epochs,log_freq,l2,lr):{grid_param}")
            except Exception as e:
                loggerg.info("Error occurred: ", e)
                grid_param, grid_acc = {'n_epochs':1,'log_freq':1,'l2':1,'lr':1},0

            if highest_acc <= grid_acc:
                highest_acc = grid_acc
                best_param = {'n_epochs':grid_param[0],'log_freq':grid_param[1],'l2':grid_param[2],'lr':grid_param[3]}
            #baysian optimization
            loggerb = logging.getLogger("Labeler.bayesianOptim")
            try:
                bayes_acc,bayes_param = self.apply_bayesianOptimization(L_train_fold , self.L_test,Y_test)
                loggerb.info(f"Max. accuracy {bayes_acc} with Parameters (n_epochs,log_freq,l2,lr):{bayes_param}")
            except Exception as e:
                loggerb.info("Error occurred: ", e)
                bayes_acc, bayes_param = 0,{'n_epochs':1,'log_freq':1,'l2':1,'lr':1}
            
            if highest_acc <= bayes_acc:
                highest_acc = bayes_acc
                best_param = {'n_epochs':bayes_param['n_epochs'],'log_freq':bayes_param['log_freq'],'l2':bayes_param['l2'],'lr':bayes_param['lr']}
            
            #save best accuracy with params and train set
            best_model_list[i] = highest_acc
            best_param_list[i] = {'n_epochs':round(best_param['n_epochs']),'log_freq':round(best_param['log_freq']),'l2':best_param['l2'],'lr':best_param['lr']}
            i += 1
        
        logger.info(f"{self.k} best Parameter settings: {best_param_list}")
        final = max(best_model_list.items(), key=operator.itemgetter(1))[0]
        final_trainset = self.L_train_list[final]
        final_param = best_param_list[final]
        final_acc = best_model_list[final]

        final_result_df = pd.DataFrame(self.evaluation_data).sort_values(by=['accuracy'], ascending=False)
        logger.info('\t'+ final_result_df.to_string().replace('\n', '\n\t')) 
        #logger.info('dataframe head - {}'.format(final_result_df.to_string()))


        logger.info(f"Best Model with {self.k}-fold Cross Validation identified: Maximum reached accuracy: {final_acc} with Parameters: {final_param}")
        self.label_model = LabelModel(cardinality = 9, verbose=False)
        if final_param['l2'] == 0: 
            final_param['l2'] = 0.1
        if final_param['lr'] == 0:
            final_param['lr'] = 0.001
        if final_param['lr'] > 0.2:
            final_param['lr'] = 0.02
        self.label_model.fit(L_train=final_trainset, n_epochs=final_param['n_epochs'], seed=123, log_freq=final_param['log_freq'], l2=final_param['l2'], lr=final_param['lr'])

    @loggingdecorator("label.function")
    def apply_randomSearch(self,train,test,y_test, max_evals = 7):       
        logger = logging.getLogger("Labeler.randomSearch")
        #random.seed(123)
        #  Choose random hyperparameters until reach max evaluations
        results ={}
        for i in range(max_evals):
            p = {'n_epochs':np.random.randint(low = 10,high = 800),
            'log_freq':np.random.randint(low = 10,high = 200),
            'l2':round(np.random.uniform(low = 0.1,high = 2.0, size = 1)[0],1),
            'lr':round(np.random.uniform(low = 0.001,high = 0.02, size = 1)[0],3)
            }
            #logger.info(f"Random choosen hyperparameters {p}")

            label_model = LabelModel(cardinality = 9, verbose=False)
            label_model.fit(L_train=train, n_epochs=p['n_epochs'], seed=123, log_freq=p['log_freq'], l2=p['l2'], lr=p['lr'])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            results[(p['n_epochs'],p['log_freq'],p['l2'],p['lr'])]=label_model_acc
            self.evaluation_data.append({'Type':'RandomSearch','n_epochs':p['n_epochs'],'log_freq':p['log_freq'],'l2':p['l2'],'lr':p['lr'],'accuracy':label_model_acc})
        
        best_parameters = max(results.items(), key=operator.itemgetter(1))[0]
        highest_accuracy = results[best_parameters]

        print(f"[Random Search]: Max. accuracy {highest_accuracy} with Parameters (n_epochs,log_freq,l2,lr): {best_parameters}")
        return best_parameters, highest_accuracy 

    @loggingdecorator("label.function")
    def apply_gridSearch(self,train,test,y_test):
        logger = logging.getLogger("Labeler.gridSearch")
        
        hyperparameter_space ={
            'n_epochs':np.arange(50, 650, 100).tolist(),
            'log_freq':np.arange(10, 220, 50).tolist(),
            'l2':np.arange(0.1, 0.6, 0.1).tolist(),
            'lr':np.arange(0.002, 0.012, 0.002).tolist()
        }
        permutations = list(product(hyperparameter_space['n_epochs'],hyperparameter_space['log_freq'],hyperparameter_space['l2'],hyperparameter_space['lr']))
        # permutations = {k: v for k, v in hyperparameter_space.items()}
        results = {}
        
        for p in permutations:
            #logger.info(f"Selected parameter {p}")
            label_model = LabelModel(cardinality = 9, verbose=False)
            label_model.fit(L_train=train, n_epochs=p[0], seed=123, log_freq=p[1], l2=p[2], lr=p[3])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            results[(p[0],p[1],p[2],p[3])]=label_model_acc
            self.evaluation_data.append({'Type':'GridSearch','n_epochs':p[0],'log_freq':p[1],'l2':p[2],'lr':p[3],'accuracy':label_model_acc})

        best_parameters = max(results.items(), key=operator.itemgetter(1))[0]
        highest_accuracy = results[best_parameters]

        print(f"[Grid Search]: Max. accuracy {highest_accuracy} with Parameters (n_epochs,log_freq,l2,lr): {best_parameters}")

        return best_parameters, highest_accuracy


    @loggingdecorator("label.function")
    def apply_bayesianOptimization(self,train,test,y_test):
        label_model = LabelModel(cardinality = 9, verbose=False)
        hyperparameter_space ={
            'n_epochs':[10,800],
            'log_freq':[10,200],
            'l2':[0,2],
            'lr':[0,1]
        }

        def model_train(n_epochs,log_freq,l2,lr):
            if l2 == 0:
                l2 = 0.1
            if lr == 0:
                lr = 0.001
            if lr > 0.2:
                lr = 0.02
            label_model.fit(L_train=train, n_epochs=round(n_epochs), seed=123, log_freq=round(log_freq), l2=l2, lr=lr)
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            self.evaluation_data.append({'Type':'BayesianOptim','n_epochs':n_epochs,'log_freq':log_freq,'l2':l2,'lr':lr,'accuracy':label_model_acc})
            return label_model_acc

        optimizer = bayes_opt.BayesianOptimization(f=model_train,pbounds =hyperparameter_space ,verbose = 2, random_state = 4)
        optimizer.maximize(init_points = 5, n_iter = 25)        

        best_parameters = optimizer.max["params"]
        highest_accuracy = optimizer.max["target"]

        print(f"[Bayesian Optimization]: Max. accuracy {highest_accuracy} with Parameters: {best_parameters}")
        return highest_accuracy,best_parameters

    def plot_bayseian(optimizer):
        plt.figure(figsize = (15, 5))
        plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
        plt.grid(True)
        plt.xlabel("Iteration", fontsize = 14)
        plt.ylabel("Black box function f(x)", fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def apply_trained_model(self):
        preds_test_label = self.label_model.predict(L=self.L_test)
        self.test_df['LABEL'] = preds_test_label
        print(self.test_df.shape)
        self.test_df = self.test_df[self.test_df['LABEL'] != 1]
        print(self.test_df.shape)
        
        # preds_val_label = self.label_model.predict(L=self.L_val)
        # self.validate_df['LABEL'] = preds_val_label
        # print(self.validate_df.shape)
        # self.validate_df = self.validate_df[self.validate_df['LABEL'] != 1]
        # print(self.validate_df.shape)    


    def show_samples_per_class(self):
        df1 = self.data.iloc[self.L_train[:, 1] == ELECTRIFICATION].sample(10, random_state=1)[['text','TOPIC']]
        #df2 = self.data.iloc[self.L_train[:, 1] == AUTONOMOUS].sample(10, random_state=1)[['text','TOPIC']]
        df3 = self.data.iloc[self.L_train[:, 1] == DIGITALISATION].sample(10, random_state=1)[['text','TOPIC']]
        #df4 = self.data.iloc[self.L_train[:, 1] == INDIVIDUALISATION].sample(10, random_state=1)[['text','TOPIC']]
        print(df1)
    
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
        #self.save_data()

if __name__ == "__main__":
    l = Labeler('de',r"files\topiced_texts.feather",r"files\labeled_texts.feather")
    test = l.data['text'].tolist()
    print(test)
    l.run()
    # e = Labeler('en',r"files\topiced_texts_en.feather",r"files\labeled_texts_en.feather")
    # e.run()
    


    


    #kmeans test
    # cluster_names = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\kMeans_cluster.feather").to_dict('records')[0]
    # kmeans = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans.pkl", 'rb'))
    # kmeans_vectoizer = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_vectorizer.pkl", 'rb')) 
    # transformed = kmeans_vectoizer.transform(["Electric"])
    # cluster = str(kmeans.predict(transformed.toarray())[0]+1)
    # value = ABSTAIN
    # list_of_values = [ABSTAIN, AUTONOMOUS,ELECTRIFICATION,CONNECTIVITY,SHARED,SUSTAINABILITY,DIGITALISATION,INDIVIDUALISATION]
    # try:
    #     value = cluster_names[str(cluster)]
    # except IndexError:
    #     value = ABSTAIN

    # for v in list_of_values:
    #     if str(value) == str(v):
    #         value = v
    # print(value)
    