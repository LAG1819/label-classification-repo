# <Automated Labeler which gets trained on extracted topics of cleaned data.>
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
import time
import numpy as np
import pickle
from sys import exit
from itertools import product
import numpy as np
import logging 
from datetime import date
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import LabelingFunction
import bayes_opt
from sklearn.model_selection import KFold

ABSTAIN = -1
AUTONOMOUS = 0
CONNECTIVITY = 1
DIGITALISATION = 2
ELECTRIFICATION = 3
INDIVIDUALISATION = 4
SHARED = 5
SUSTAINABILITY = 6

#DEFINE CLUSTERING MODEL (SIMPLE) K-Means###
#SELECT ALL KEYWORDS BASED ON SUFFICENT SOURCES INCLUDING LEVENSTHEIN DISTANCE###
#######################################################################################################################################################
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

def kMeans_cluster(x, label, kmeans, kmeans_vectorizer):    
    text = kmeans_vectorizer.transform([str(x)]).toarray()
    cluster = kmeans.predict(text)[0]

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
kmeans_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_de.pkl", 'rb')) 
kmeans_vectorizer_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_vectorizer_de.pkl", 'rb')) 
autonomous_cluster_d =  make_cluster_lf(label = AUTONOMOUS, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
electrification_cluster_d =  make_cluster_lf(label = ELECTRIFICATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
digitalisation_cluster_d =  make_cluster_lf(label = DIGITALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
connectivity_cluster_d =  make_cluster_lf(label = CONNECTIVITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
sustainability_cluster_d =  make_cluster_lf(label = SUSTAINABILITY, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
individualisation_cluster_d = make_cluster_lf(label = INDIVIDUALISATION, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)
shared_cluster_d = make_cluster_lf(label = SHARED, kmeans= kmeans_de, kmeans_vectorizer= kmeans_vectorizer_de)

#load trained kMeans, fitted vectorizer for english kMeans
kmeans_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_en.pkl", 'rb')) 
kmeans_vectorizer_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_vectorizer_en.pkl", 'rb')) 
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
        if (logger.hasHandlers()):
            logger.handlers.clear()

        handler  = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        filenames =  str(os.path.dirname(__file__)).split("src")[0] + r'models\label\model_tuning_'+lang+r'\automated_labeling_'+lang+r'.log'
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
                # autonomous_cluster_d,
                # electrification_cluster_d,
                # digitalisation_cluster_d,
                # connectivity_cluster_d, 
                # sustainability_cluster_d,
                # individualisation_cluster_d, 
                # shared_cluster_d,
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
        self.update_data = True
        self.update_model = False
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = column
        self.data = self.load_data()
        self.train_df, self.validate_df, self.test_df, self.train_test_df = self.generate_trainTestdata(lang)
        self.evaluation_data = []

        ##Start labeling##
        logger.info("Automated Labeling started with Language {l}, Text-Column: {t_col} and data file {path} (source) created. Target file is {tpath}".format(l = lang, path = s_path, tpath = t_path, t_col = self.text_col))
        self.run()
    
    def load_data(self) -> pd.DataFrame:
        """Loads cleaned dataset containing topics as well.

        Returns:
            pd.DataFrame: Returns pandas DataFrame containing the cleaned texts and its topics
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path)
        return df.replace(np.nan, "",regex = False)

    def generate_trainTestdata(self, lang:str) -> pd.DataFrame:
        test_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_testset_"+lang+r".xlsx"
        train_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_trainset_"+lang+r".feather"
        val_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_valset_"+lang+r".xlsx"

        if os.path.exists(test_path):
            test = pd.read_excel(test_path, index_col = 0)
            test = test[test['LABEL']!= -2]

            validate = pd.read_excel(val_path, index_col = 0)
            validate = validate[validate['LABEL']!= -2]

            train = pd.read_feather(train_path)     
        else:
            self.data['LABEL'] = -2
            #60 % trainset, 20% testset, 20% validation set
            train, validate, test = np.split(self.data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(self.data)), int(.8*len(self.data))])
            
            dfs = [train, test, validate]
            dfs = [df.reset_index(drop = True, inplace = True) for df in dfs]
            
            train.to_feather(train_path)
            test.to_excel(test_path)
            validate.to_excel(val_path)
            
            logger = logging.getLogger("Labeler")
            logger.info("Train, Test and Validate Dataset were generated. Please label train and validate data before further proceeding!")
            exit("No labeled Test and Validate data exist!")
        
        train_copy = train.copy()
        train_copy['LABEL'] = -2
        train_test = pd.concat([train_copy,test])
        train_test.reset_index(inplace = True)
        
        dfs = [train, test, validate, train_test,self.data]
        dfs = [df.rename(columns = {self.text_col : 'text'},inplace=True) for df in dfs]

        print(train.shape,test.shape,validate.shape)

        return train, validate, test, train_test

    def train_labeling_functions(self):
        """First step of Snorkel. Previous defined Labeling Functions are applied to the loaded Training, Test and Validation Datasets.
        """
        logger = logging.getLogger("Labeler")
        logger.info("Application of Labeling Functions: {f}".format(f = self.lfs))
        self.applier = PandasLFApplier(lfs=self.lfs)
        try:
            #k_fold Cross Validation
            self.L_train_list =[]
            for j in range(2,3):
                k_fold = KFold(n_splits = j,shuffle = True, random_state = 12)
                i = 1
                for split in k_fold.split(self.train_test_df):
                    logger.info(f"Training of {j}-Fold Cross-Validation with Trainingsplit {i} started.")
                    fold_train_df = self.train_test_df.iloc[split[0]]
                    
                    fold_test_df = self.train_test_df.iloc[split[1]]
                    fold_test_df = fold_test_df[fold_test_df['LABEL']!= -2]
                    y_test = fold_test_df['LABEL'].to_numpy()

                    l_train = self.applier.apply(df=fold_train_df)
                    l_test = self.applier.apply(df=fold_test_df)
                    self.L_train_list.append((l_train,l_test,y_test,j,i))

                    # polarity = The set of unique labels this LF outputs (excluding abstains)
                    # coverage = percentage of objects the LF labels
                    # overlap  = percentage of objects with more than one label. 
                    # conflict = percentage of objects with conflicting labels.
                    L_analyis = LFAnalysis(L=l_train, lfs=self.lfs)

                    logger.info(f"Training set coverage of {j}-Fold Cross Validation with Trainingset {i}: {100 * L_analyis.label_coverage(): 0.1f}%")
                    self.analysis_training_result(l_train,j,i,L_analyis.lf_summary())

                    self.train_model(l_train,l_test,y_test,j,i)
                    logger.info(f"{j}-Fold Cross-Validation with Trainingsplit {i} were trained.")

                    result_df = pd.DataFrame(self.evaluation_data).sort_values(by=['accuracy'], ascending=[False])
                    current_best = result_df.to_dict('records')[0]
                    logger.info(f"Current best training: {current_best}")
                    i+=1
            
            ##save evaluation results##
            final_result_df = pd.DataFrame(self.evaluation_data).sort_values(by=['accuracy'], ascending=[False])
            self.save_results(final_result_df)

            ##get best parameter settings and associated model##
            final_param = final_result_df.to_dict('records')
            final_param = final_param[0]            
            logger.info(f"Best Model with {final_param['k-fold']}-fold Cross Validation identified. \n Maximum reached accuracy: {final_param['accuracy']}. \n Path: {final_param['model']}")

            self.label_model = LabelModel(cardinality = 7, verbose=False)
            self.label_model.load(final_param['model'])
        except KeyboardInterrupt:
            ##save evaluation results##
            final_result_df = pd.DataFrame(self.evaluation_data).sort_values(by=['accuracy'], ascending=[False])
            self.save_results(final_result_df)
            
            ##get best parameter settings and associated model##
            final_param = final_result_df.to_dict('records')
            final_param = final_param[0]
            logger.info(f"Best Model with {final_param['k-fold']}-fold Cross Validation identified. \n Maximum reached accuracy: {final_param['accuracy']}. \n Path: {final_param['model']}")

            self.label_model = LabelModel(cardinality = 7, verbose=False)
            self.label_model.load(final_param['model'])
            ##test and save model##
            self.test_model()
            self.save_model()
            return

    def analysis_training_result(self, l_train, k, i,L_analyis):
        """Analysation of applied Labeling Functions and the coverage (in %) of each Function on the dataset.
        """
        logger = logging.getLogger("Labeler")
        logger.info('Validate metrics of trained Labeling Functions.')

        print(L_analyis)
        
        logger.info(f"Validate metrics of {k}-fold Cross Validation with Trainingset {i}")
        # autonomous_cl, electrification_cl,digitalisation_cl,connectivity_cl, sustainability_cl,individualisation_cl, shared_cl,\
        autonomous_k,electrification_k,digitalisation_k,connectivity_k,sustainability_k,individualisation_k,shared_k = (l_train != ABSTAIN).mean(axis=0)
        
        coverage = [
            # ["cluster","autonomous", autonomous_cl * 100],
            #         ["cluster","electrification", electrification_cl * 100],
            #         ["cluster","digitalisation",digitalisation_cl * 100],
            #         ["cluster","connectivity",connectivity_cl * 100],
            #         ["cluster","sustainability",sustainability_cl * 100],
            #         ["cluster","individualisation",individualisation_cl*100],
            #         ["cluster","shared", shared_cl * 100],
                    ["keyword","autonomous", autonomous_k * 100],
                    ["keyword","digitalisation",digitalisation_k * 100],
                    ["keyword","electrification",electrification_k * 100],
                    ["keyword","sustainability",sustainability_k * 100],
                    ["keyword","connectivity",connectivity_k * 100],
                    ["keyword","individualisation",individualisation_k * 100],
                    ["keyword","shared",shared_k * 100]
                    ]
        #get current trial number
        eval_path = r'models\label\model_tuning_'+self.lang+r'\eval_results.feather'
        path = str(os.path.dirname(__file__)).split("src")[0]
        ##check if target_path already exists
        if os.path.exists(path+eval_path):
            df_all = pd.read_feather(path+eval_path)
            trial = df_all[['Trial']].sort_values(by=['Trial'], ascending=[False]).to_dict('records')[0]['Trial']
            trial = trial+1
        else:
            trial = 0

        #save coverage data next to eval data
        coverage_path = r'models\label\model_tuning_'+self.lang+r'\coverage_results.feather'
        coverage_df = pd.DataFrame(data = coverage, columns = ["LF","CLASS","COVERAGE"])

        coverage_df['TRIAL'] = trial
        coverage_df['k_fold'] = k
        coverage_df['k_fold_split'] = i
        
        if os.path.exists(path+coverage_path):
            all_coverage_df = pd.read_feather(path+coverage_path)
            coverage_df_all = pd.concat([all_coverage_df,coverage_df])
            coverage_df_all.reset_index(inplace=True, drop = True)
            coverage_df_all.to_feather(path+coverage_path)
        else:
            coverage_df.reset_index(inplace=True, drop = True)
            coverage_df.to_feather(path+coverage_path)

    def train_model(self,l_train,l_test,y_test,j,i):
        """Evaluation of the best parameters for Snorkels Labeling Model by (Hyper-)parameter Tuning.
        Selected optimizer are: Grid Search, Random Search and Bayesian Optimization. 
        """    
        # for trainset in self.L_train_list:
        #     L_train_fold, L_test_fold,Y_test, k, i = trainset[0],trainset[1],trainset[2],trainset[3],trainset[4]
        L_train_fold, L_test_fold,Y_test,k,i = l_train,l_test,y_test,j,i
        logger = logging.getLogger("Labeler")
        logger.info(f"Training and Evaluation of best Model with {k}-fold Cross Validation with Trainingset {i}")

        #random search
        loggerr = logging.getLogger("Labeler.randomSearch")
        try:
            self.apply_randomSearch(L_train_fold,L_test_fold,Y_test,k,i)
        except AttributeError:
            pass
        except Exception as e:
            loggerr.warning("Error occurred: ", e)
            pass
        
        # # #grid search
        # loggerg = logging.getLogger("Labeler.gridSearch")
        # try:
        #     self.apply_gridSearch(L_train_fold,L_test_fold,Y_test,k,i)
        # except AttributeError:
        #     pass
        # except Exception as e:
        #     loggerg.warning("Error occurred: ", e)
        #     pass

        # #bayesian search
        # loggerb = logging.getLogger("Labeler.bayesianOptim")
        # try:
        #     self.apply_bayesianOptimization(L_train_fold,L_test_fold,Y_test,k,i)
        # except AttributeError:
        #     pass
        # except Exception as e:
        #     loggerb.warning("Error occurred: ", e)
        #     pass

    @loggingdecorator("label.function")
    def apply_randomSearch(self,train,test,y_test,k,i, max_evals = 20):
        
        # Set up random search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.lang+r"\random_search_"+self.text_col
        
        #  random.seed(123)
        #  Choose random hyperparameters until reach max evaluations
        eval_data_intern = []
        for l in range(max_evals):
            p = {'n_epochs':np.random.randint(low = 10,high = 800),
            'log_freq':np.random.randint(low = 10,high = 200),
            'l2':round(np.random.uniform(low = 0.1,high = 2.0, size = 1)[0],1),
            'lr':round(np.random.uniform(low = 0.001,high = 0.02, size = 1)[0],3),
            'optimizer':np.random.choice(["sgd", "adam", "adamax"],1)[0],
            'lr_scheduler': np.random.choice(["constant", "linear", "exponential", "step"],1)[0]
            }
            label_model = LabelModel(cardinality = 7, verbose=False)
            label_model.fit(L_train=train, n_epochs=p['n_epochs'], seed=123, log_freq=p['log_freq'], l2=p['l2'], lr=p['lr'], optimizer = p['optimizer'], lr_scheduler = p['lr_scheduler'])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            
            eval_data_intern.append({'Type':'RandomSearch','n_epochs':p['n_epochs'],'log_freq':p['log_freq'],'l2':p['l2'],'lr':p['lr'],'optimizer':p['optimizer'],'lr_scheduler':p['lr_scheduler'],'accuracy':label_model_acc,'k-fold':k,'trainingset':i})
        
        # sort results by accuracy
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]
            
        # Set up model checkpoint folder and save model including log
        rand_name = str(time.time())[-5:] 
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)
        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'], lr_scheduler = best_model['lr_scheduler'])
        label_model.save(model_folder+r"\label_model.pkl")

        # add best result to overall evaluation results
        self.evaluation_data.append({'Type':'RandomSearch','n_epochs':best_model['n_epochs'],'log_freq':best_model['log_freq'],'l2':best_model['l2'],'lr':best_model['lr'],'optimizer':best_model['optimizer'],'lr_scheduler':best_model['lr_scheduler'],'accuracy':best_model['accuracy'],'k-fold':k,'trainingset':i, 'model':model_folder+r"\label_model.pkl"})

    @loggingdecorator("label.function")
    def apply_gridSearch(self,train,test,y_test,k,i):

        # Set up grid search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.lang+r"\grid_search_"+self.text_col
        
        hyperparameter_space ={
            'n_epochs':[10,30,50,100],#np.arange(10, 100, 20).tolist(),
            'log_freq':[30,90,170],#np.arange(10, 220, 50).tolist(),
            'l2':[0.1,0.2,0.3],#np.arange(0.1, 0.6, 0.1).tolist(),
            'lr':[0.00001,0.0001, 0.001],#np.arange(0.0001, 0.01, 0.002).tolist(), 
            'optimizer':["sgd", "adam", "adamax"],
            'lr_scheduler': ["constant", "linear", "exponential", "step"]
        }
        permutations = list(product(hyperparameter_space['n_epochs'],hyperparameter_space['log_freq'],hyperparameter_space['l2'],hyperparameter_space['lr'],hyperparameter_space['optimizer'], hyperparameter_space['lr_scheduler']))
        eval_data_intern = []
        for p in permutations:
            #logger.info(f"Selected parameter {p}")
            label_model = LabelModel(cardinality = 7, verbose=False)
            label_model.fit(L_train=train, n_epochs=p[0], seed=123, log_freq=p[1], l2=p[2], lr=p[3], optimizer = p[4], lr_scheduler = p[5])
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            
            eval_data_intern.append({'Type':'GridSearch','n_epochs':p[0],'log_freq':p[1],'l2':p[2],'lr':p[3],'optimizer':p[4],'lr_scheduler':p[5],'accuracy':label_model_acc,'k-fold':k,'trainingset':i})
            
        # sort results by accuracy
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]

        # Set up model checkpoint folder and save model including log
        rand_name = str(time.time())[-5:]
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)

        # add best result to overall evaluation results
        self.evaluation_data.append({'Type':'GridSearch','n_epochs':best_model['n_epochs'],'log_freq':best_model['log_freq'],'l2':best_model['l2'],'lr':best_model['lr'],'optimizer':best_model['optimizer'],'lr_scheduler':best_model['lr_scheduler'],'accuracy':best_model['accuracy'],'k-fold':k,'trainingset':i, 'model':model_folder+r"\label_model.pkl"})

        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'], lr_scheduler = best_model['lr_scheduler'])
        label_model.save(model_folder+r"\label_model.pkl")


    @loggingdecorator("label.function")
    def apply_bayesianOptimization(self,train,test,y_test,k,i, max_evals = 40):

        # Set up beysian optimization search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.lang+r"\bayes_search_"+self.text_col

        hyperparameter_space ={
            'n_epochs':[10,150],
            'log_freq':[10,200],
            'l2':[0.1,2.0],
            'lr':[0.0001,0.01],
            'optimizer':[0,2],
            #'lr_scheduler': [0,3]
        }
        eval_data_intern = []
        def model_train(n_epochs,log_freq,l2,lr, optimizer):
            label_model = LabelModel(cardinality = 7, verbose=False)
            if round(optimizer) == 0:
                optim = "sgd"
            elif round(optimizer) == 1:
                optim = "adam"
            elif round(optimizer) == 2:
                optim = "adamax"
            else:
                optim = "adam"

            # if round(lr_scheduler) == 0:
            #     lr_schedul = "constant" 
            # elif round(lr_scheduler) == 1:
            #     lr_schedul = "linear"
            # elif round(lr_scheduler) == 2:
            #     lr_schedul = "exponential"
            # elif round(lr_scheduler) == 3:
            #     lr_schedul = "step"
            # else:
            #     lr_schedul = "step"

            label_model.fit(L_train=train, n_epochs=round(n_epochs), seed=123, log_freq=round(log_freq), l2=l2, lr=lr, optimizer = optim)#, lr_scheduler = lr_schedul)
            label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")["accuracy" ]
            
            # Save evaluation results temporarily
            eval_data_intern.append({'Type':'BayesSearch','n_epochs':round(n_epochs),'log_freq':round(log_freq),'l2':l2,'lr':lr,'optimizer':optim,'lr_scheduler':'None','accuracy':label_model_acc,'k-fold':k,'trainingset':i})
            return label_model_acc

        optimizer = bayes_opt.BayesianOptimization(f=model_train,pbounds =hyperparameter_space ,verbose = 1, random_state = 4)
        optimizer.maximize(init_points = 5, n_iter = max_evals)        

        # best_parameters = optimizer.max["params"]
        # highest_accuracy = optimizer.max["target"]

        # sort results by accuracy
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]

        # Set up model checkpoint folder and save model including log
        rand_name = str(time.time())[-5:] 
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)

        # add best result to overall evaluation results
        self.evaluation_data.append({'Type':'BayesSearch','n_epochs':best_model['n_epochs'],'log_freq':best_model['log_freq'],'l2':best_model['l2'],'lr':best_model['lr'],'optimizer':best_model['optimizer'],'lr_scheduler':'None','accuracy':best_model['accuracy'],'k-fold':k,'trainingset':i, 'model':model_folder+r"\label_model.pkl"})

        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'])
        label_model.save(model_folder+r"\label_model.pkl")
        
        
    def test_model(self):
        logger = logging.getLogger("Labeler")
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(self.lang)+"_"+self.text_col+r".pkl"
        
        Y_val = self.validate_df['LABEL'].to_numpy()
        L_val = self.applier.apply(df=self.validate_df)
        preds_val_label = self.label_model.predict(L=L_val)
        
        validate_acc = self.label_model.score(L=L_val, Y=Y_val, tie_break_policy="random")["accuracy"]
        logger.info(f"Accuracy on Validation set: {validate_acc}")
        if os.path.exists(path):
            old_label_model = LabelModel(cardinality = 7, verbose=False)            
            old_label_model.load(path)
            old_model_acc = old_label_model.score(L=L_val, Y=Y_val, tie_break_policy="random")["accuracy"] 

            if validate_acc > old_model_acc:
                self.update_model = True
                self.update_data = True
                logger.info(f"New model accuracy: {validate_acc}. Saved model accuracy: {old_model_acc} (Applied on Validation Set)\nModel will be overwritten.")
            else:
                self.update_model = False
                self.update_data = False
                logger.info(f"New model accuracy: {validate_acc}. Saved model accuracy: {old_model_acc} (Applied on Validation Set)\nModel will not be overwritten.")
        else:
            self.update_model = True
            self.update_data = True            
            logger.info(f"Model accuracy: {validate_acc}. No existing Model. Model will be saved.")
        
        validation_df = self.validate_df
        validation_df['LABEL'] = preds_val_label
        before_val_shape = self.validate_df.shape
        validation_df = validation_df[validation_df['LABEL'] != -1]
        after_val_shape = validation_df.shape
        print(f"Shape of Validationset before: {before_val_shape} and after: {after_val_shape}")
    
    def save_model(self):
        """Saving of trained Label model as pickle file with optimized parameter.
        """
        logger = logging.getLogger("Labeler")
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(self.lang)+"_"+self.text_col+r".pkl"

        if self.update_model:
            self.label_model.save(path)
            logger.info("Model saved!")
        else:
            logger.info(f"Model not saved!")
            self.update_data = False
            # self.label_model.save(path)
            # self.update_data = True   

    def apply_model(self):
        logger = logging.getLogger("Labeler")
        if self.update_data:
            
            logger.info(f"Language:{self.lang}. Applying trained model with best parameters on whole data set.")

            self.L_data = self.applier.apply(df= self.data)
            label = self.label_model.predict(L=self.L_data)

            self.data['LABEL'] = label
            before_data_shape = self.data.shape
            self.data = self.data[self.data['LABEL'] != -1]
            after_data_shape = self.data.shape
            
            logger.info(f"Shape of Dataset before: {before_data_shape} and after: {after_data_shape}")
        else:
            logger.info(f"Language:{self.lang}. NOT Applying trained model with best parameters and trainset on whole data set due to worse model accuracy.")

    def save_data(self):
        """Saves labeled data as feather into files folder.
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        logger = logging.getLogger("Labeler")
        
        self.data.reset_index(inplace=True)
        self.data.rename(columns = {'text':self.text_col},inplace=True)

        if self.update_data:
            self.data.to_feather(path)
            logger.info("Data with applied labels saved!")
        else:
            logger.info("Data with applied labels not saved!")

    def save_results(self, df_new):
        t_path = r'models\label\model_tuning_'+self.lang+r'\eval_results.feather'
        path = str(os.path.dirname(__file__)).split("src")[0]
        trial = 0
        ##check if target_path already exists
        if os.path.exists(path+t_path):
            df_all = pd.read_feather(path+t_path)

            trial = df_all[['Trial']].sort_values(by=['Trial'], ascending=[False]).to_dict('records')[0]['Trial']
            df_new['Trial'] = trial+1
            
            df_all_new = pd.concat([df_all,df_new])
        else:
            df_all_new = df_new
            df_all_new['Trial'] = trial

        df_all_new['TEXT'] = self.text_col
        df_all_new.reset_index(inplace = True, drop = True)
        df_all_new.to_feather(path+t_path)

            
    def run(self):
        self.train_labeling_functions()
        #self.test_model()
        #self.save_model()
        #self.apply_model()
        #self.save_data()

if __name__ == "__main__":
    for lang in ['de']:#,'en'
        topic_labeling = Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_TOPIC'+".feather",'TOPIC')
        # text_labeling = Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_URL_TEXT'+".feather",'URL_TEXT')
   
    ####test of model loading###
    # path =r'D:\University\Hochschule der Medien_M.Sc. Data Science\Master\Repository\ml-classification-repo\models\label\model_tuning_de\grid_search\model_10857_n_epochs_50_log_freq_10_l2_0.1_lr_0.002_optimizer_adam_2023-02-09\label_model.pkl'
    # path2 = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(lang)+r".pkl"
    # model = LabelModel(cardinality = 7, verbose=False)
    # model2 = LabelModel(cardinality = 7,verbose = False)
    # model.load(path)
    # model2.load(path2)
    # lfs = [
    #             autonomous_keywords,
    #             electrification_keywords,
    #             digitalisation_keywords,
    #             connectivity_keywords,
    #             sustainability_keywords,
    #             individualisation_keywords,
    #             shared_keywords,
    #             ]
    # applier = PandasLFApplier(lfs=lfs)
    # Y_test = topic_labeling.validate_df['LABEL'].to_numpy()
    # L_test = applier.apply(df=topic_labeling.validate_df)
    # acc = model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy" ] 
    # acc2 = model2.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy" ] 
    # print(acc,acc2)
    # print("FIN")

