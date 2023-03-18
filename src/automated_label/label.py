# <Automated Labeling of texts. Label Model trained seperately on topics or text in german or english for comparison. This is process step 4.>
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
import random
import pickle
from sys import exit
from itertools import product
import numpy as np
import logging 
from datetime import date
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.apply.spark import SparkLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling import LabelingFunction
import bayes_opt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
def loggingdecorator(name):
    """Helper function for logging. Used to make logging in a lucid way.

    Args:
        name (_type_): Name to show log with.

    Returns:
        _type_: Returns arranged logging. 
    """
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

############################################################## USER-DEFINED LABELING FUNCTIONS ######################################################################## 
    
def kMeans_cluster(x:str, label:int, kmeans:KMeans, kmeans_vectorizer:TfidfVectorizer,ABSTAIN:int) -> int:
    """K-Means clustering with loaded and pre-trained KMeans cluster of an input sentence.
    This function is only used if labeling of total dataset is required.

    Args:
        x (str): String to cluster.
        label (int): Input label to compare cluster of clustered sentence with.
        kmeans (KMeans): Pre-trained KMeans cluster.
        kmeans_vectorizer (TfidfVectorizer): Associated TfidfVectorizer of pre-trained KMeans cluster.
        ABSTAIN (int): Default if label can not be assigned.

    Returns:
        int: Returns -1 (ABSTAIN) if predicted cluster is not dedicated label. Otherwise returns label.
    """
    text = kmeans_vectorizer.transform([str(x)]).toarray()
    cluster = kmeans.predict(text)[0]

    if int(cluster) == int(label):
        return label
    else:
        return ABSTAIN
    

def make_cluster_lf(label:int, kmeans:KMeans, kmeans_vectorizer:TfidfVectorizer, abstain:int) -> LabelingFunction:
    """Generate a Label Function (LF) based on an input K-Means cluster and associated TfidfVectorizer.
    This Labeling Function is only used if labeling of total dataset is required.

    Args:
        label (int): Selected label to which the LF should assign to.
        kmeans (KMeans): Pre-trained KMeans cluster.
        kmeans_vectorizer (TfidfVectorizer): Associated TfidfVectorizer of pre-trained KMeans cluster.
        abstain (int): Default if label can not be assigned.

    Returns:
        LabelingFunction: Returns generated Labeling Function (LF).
    """
    return LabelingFunction(
        name=f"cluster_{str(label)}",
        f=kMeans_cluster,
        resources=dict(label=label, kmeans =kmeans, kmeans_vectorizer=kmeans_vectorizer, ABSTAIN=abstain),
    )

def keyword_lookup(x:str, keywords:list, label:int, ABSTAIN:int)->int:
    """Keyword Matching of an input sentence. Keywords can be defined in Seed.xlsx.
    This function is used in both cases: if labeling of total or partial dataset is required.

    Args:
        x (str): String to assign keyword matching to.
        keywords (list): List of dedicated keywords to match with.
        label (int): Input label associated with the keywords.
        ABSTAIN (int): Default if label can not be assigned.

    Returns:
        int: Returns -1 (ABSTAIN) if no matched keyword found in sentence. Otherwise returns label.
    """
    if any(word in x.text.lower() for word in keywords):
        return label
    else:
        return ABSTAIN

def make_keyword_lf(keywords:list, label:int, abstain:int) -> LabelingFunction:
    """Generate a Label Function (LF) based on keyword_lookup function. Keywords can be defined in Seed.xlsx.
    This Labeling Function is used in both cases: if labeling of total or partial dataset is required.

    Args:
        keywords (list):  List of dedicated keywords associated to label.
        label (int): Selected label to which the LF should assign to.
        abstain (int): Default if label can not be assigned.

    Returns:
        LabelingFunction: Returns generated Labeling Function (LF).
    """
    return LabelingFunction(
        name=f"keyword_{label}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label, ABSTAIN=abstain),
    )

############################################################## Label Model and Training ################################################################################## 
class Labeler:
    """Class to train and apply a Label Model of Snorkel on data.
    """

    #USER DEFINED CLASS-LABEL
    #Note: Class ABSTAIN is obligatory, do not change or otherwise the Label Model is not working correctly!
    __ABSTAIN = -1
    AUTONOMOUS = 0
    CONNECTIVITY = 1
    DIGITALISATION = 2
    ELECTRIFICATION = 3
    INDIVIDUALISATION = 4
    SHARED = 5
    SUSTAINABILITY = 6

    #Generation of all Labeling Functions (LF) for Keyword Matching
    __absolute_path = str(os.path.dirname(__file__)).split("src")[0] + r"files/Seed.xlsx"
    __seed_data = pd.read_excel(__absolute_path,header = 0) 
    __df = __seed_data[['AUTONOMOUS','ELECTRIFICATION','CONNECTIVITY','SHARED','SUSTAINABILITY','DIGITALISATION','INDIVIDUALISATION']]
    __autonomous_keywords = make_keyword_lf(keywords = __df['AUTONOMOUS'].dropna().tolist(), label = AUTONOMOUS, abstain = __ABSTAIN)
    __electrification_keywords = make_keyword_lf(keywords = __df['ELECTRIFICATION'].dropna().tolist(), label = ELECTRIFICATION, abstain = __ABSTAIN)
    __digitalisation_keywords = make_keyword_lf(keywords = __df['DIGITALISATION'].dropna().tolist(), label = DIGITALISATION, abstain = __ABSTAIN)
    __connectivity_keywords = make_keyword_lf(keywords = __df['CONNECTIVITY'].dropna().tolist(), label = CONNECTIVITY, abstain = __ABSTAIN)
    __sustainability_keywords = make_keyword_lf(keywords = __df['SUSTAINABILITY'].dropna().tolist(), label = SUSTAINABILITY, abstain = __ABSTAIN)
    __individualisation_keywords = make_keyword_lf(keywords = __df['INDIVIDUALISATION'].dropna().tolist(), label = INDIVIDUALISATION, abstain = __ABSTAIN)
    __shared_keywords = make_keyword_lf(keywords = __df['SHARED'].dropna().tolist(), label = SHARED, abstain = __ABSTAIN)

    #load trained kMeans, fitted vectorizer for german kMeans
    _kmeans_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_de.pkl", 'rb')) 
    _kmeans_vectorizer_de = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_vectorizer_de.pkl", 'rb')) 
    #Generation of german Labeling Functions (LF) for Clustering
    __autonomous_cluster_d =  make_cluster_lf(label = AUTONOMOUS, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __electrification_cluster_d =  make_cluster_lf(label = ELECTRIFICATION, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __digitalisation_cluster_d =  make_cluster_lf(label = DIGITALISATION, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __connectivity_cluster_d =  make_cluster_lf(label = CONNECTIVITY, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __sustainability_cluster_d =  make_cluster_lf(label = SUSTAINABILITY, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __individualisation_cluster_d = make_cluster_lf(label = INDIVIDUALISATION, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)
    __shared_cluster_d = make_cluster_lf(label = SHARED, kmeans= _kmeans_de, kmeans_vectorizer= _kmeans_vectorizer_de, abstain = __ABSTAIN)

    #load trained kMeans, fitted vectorizer for english kMeans
    _kmeans_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_en.pkl", 'rb')) 
    _kmeans_vectorizer_en = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/label/k_Means/kmeans_vectorizer_en.pkl", 'rb'))
    #Generation of english Labeling Functions (LF) for Clustering
    __autonomous_cluster_e =  make_cluster_lf(label = AUTONOMOUS, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __electrification_cluster_e =  make_cluster_lf(label = ELECTRIFICATION, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __digitalisation_cluster_e =  make_cluster_lf(label = DIGITALISATION, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __connectivity_cluster_e =  make_cluster_lf(label = CONNECTIVITY, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __sustainability_cluster_e =  make_cluster_lf(label = SUSTAINABILITY, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __individualisation_cluster_e = make_cluster_lf(label = INDIVIDUALISATION, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)
    __shared_cluster_e = make_cluster_lf(label = SHARED, kmeans= _kmeans_en, kmeans_vectorizer= _kmeans_vectorizer_en, abstain = __ABSTAIN)

    

    def __init__(self,lang:str,s_path:str, t_path:str, column:str, partial:bool):
        """Initialisation for Label Model training and application.

        Args:
            lang (str): Unicode of language to train model with. It can be choosen between de (german) and en (englisch).
            s_path (str): source path to file containing raw texts to clean
            t_path (str): target path to save file with cleaned texts
            column (str): Selected Column on which the data get labeled on. It can be choosen between TOPIC or URL_TEXT.
            partial (bool): Selected type of data labeling. If True a partial data labeling is applied. If False a total data labeling is applied with help of a pretrained k-Means cluster.
        """
        # Create logger and assign handler
        logger = logging.getLogger("Labeler")
        if (logger.hasHandlers()):
            logger.handlers.clear()

        handler  = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        __filenames =  str(os.path.dirname(__file__)).split("src")[0] + r'models\label\model_tuning_'+lang+r'\automated_labeling_'+lang+r'.log'
        fh = logging.FileHandler(filename=__filenames)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s]%(levelname)s|%(name)s|%(message)s"))
        logger.addHandler(fh)

        logger = logging.getLogger("Labeler.randomSearch")
        logger.setLevel(logging.INFO)
        logger = logging.getLogger("Labeler.gridSearch")
        logger.setLevel(logging.INFO)
        logger = logging.getLogger("Labeler.bayesianOptim")
        logger.setLevel(logging.INFO)

        self.lfs = [self.__autonomous_keywords, self.__electrification_keywords, self.__digitalisation_keywords, self.__connectivity_keywords, self.__sustainability_keywords,\
                    self.__individualisation_keywords, self.__shared_keywords]
        if partial == False:
            if lang == 'de':
                self.lfs = self.lfs + [self.__autonomous_cluster_d, self.__electrification_cluster_d, self.__digitalisation_cluster_d, self.__connectivity_cluster_d, self.__sustainability_cluster_d,\
                                       self.__individualisation_cluster_d, self.__shared_cluster_d]
            if lang == 'en':
                self.lfs = self.lfs + [self.__autonomous_cluster_e, self.__electrification_cluster_e, self.__digitalisation_cluster_e, self.__connectivity_cluster_e, self.__sustainability_cluster_e,\
                                       self.__individualisation_cluster_e, self.__shared_cluster_e]

         
        logger = logging.getLogger("Labeler")
        self.__lang = lang
        self.__update_data = True
        self.__update_model = False
        # self.L_train = None
        self.__label_model = None
        self.source_path = s_path
        self.__target_path = t_path
        self.__text_col = column
        self.__data = self.load_data()
        self.__train_df, self.__validate_df, self.__test_df, self.__train_test_df = self.__generate_trainTestdata(lang)
        self.__trial = self.__get_trial()

    @classmethod
    def load_data(self) -> pd.DataFrame:
        """Loads cleaned dataset containing topics as well.

        Returns:
            pd.DataFrame: Returns pandas DataFrame containing the dataset containing cleaned texts and extracted topics.
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path)
        return df.replace(np.nan, "",regex = False)
    
    @classmethod
    def __generate_trainTestdata(self, lang:str) -> pd.DataFrame:
        """Loading the training, test and validation data set if they already exist. 
        Otherwise the total dataset will be split into training, test and validation datasets. The generated test and validation dataset must then first be labeled by a domain expert!

        Args:
            lang (str): Unicode of language to train model with. It can be choosen between de (german) and en (englisch).

        Returns:
            pd.DataFrame: Returns training test and validation data set as well as training and validation data set combined.
        """
        test_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_testset_"+lang+r".xlsx"
        train_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_trainset_"+lang+r".feather"
        val_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\03_label\label_valset_"+lang+r".xlsx"

        #check if manual labeled test and validationssets exist
        if os.path.exists(test_path):
            test = pd.read_excel(test_path, index_col = 0)
            test = test[test['LABEL']!= -2]

            validate = pd.read_excel(val_path, index_col = 0)
            validate = validate[validate['LABEL']!= -2]

            train = pd.read_feather(train_path)     
        #if no manual labeled test and validationssets exist assign label column with fixed number -2
        else:
            self.__data['LABEL'] = -2
            #60 % trainset, 20% testset, 20% validation set
            train, validate, test = np.split(self.__data.sample(frac=1, random_state=42, axis = 0, replace = False),[int(.6*len(self.__data)), int(.8*len(self.__data))])
            
            dfs = [train, test, validate]
            dfs = [df.reset_index(drop = True, inplace = True) for df in dfs]
            
            train.to_feather(train_path)
            test.to_excel(test_path)
            validate.to_excel(val_path)
            
            logger = logging.getLogger("Labeler")
            logger.info("Train, Test and Validate Dataset were generated. Please label train and validate data before further proceeding!")
            exit("No labeled Test and Validate data exist! Please label generated train and test data file, stored in files/03_label/")
        
        train_copy = train.copy()
        train_copy['LABEL'] = -2
        train_test = pd.concat([train_copy,test])
        train_test.reset_index(inplace = True)
        
        dfs = [train, test, validate, train_test,self.__data]
        dfs = [df.rename(columns = {self.__text_col : 'text'},inplace=True) for df in dfs]

        print(train.shape,test.shape,validate.shape)

        return train, validate, test, train_test
    
    @classmethod
    def __train_labeling_functions(self):
        """Overall Train function of model containing k-Fold Cross Validation. Labeling Functions (LF) are applied to k-fold training and testset.
        If process got interrupted by user the best model including the evaluation results of the training up to this point will be saved anyway 
        as well as the labeling of the whole data set with the best model at that point will be applied and saved.
        """
        logger = logging.getLogger("Labeler")
        logger.info("Application of Labeling Functions: {f}".format(f = self.lfs))
        self.applier = PandasLFApplier(lfs=self.lfs)
        
        """Alternative approach of LF application with Snorkel SparkLFApplier for a faster application. 
        Due to various error issues on the part of Snorkel no further utilization of this variant."""
        # self.applier2 = SparkLFApplier(lfs=self.lfs)
        # spark = SparkSession.builder\
        # .appName("Test application")\
        #     .master("local[1]") \
        #         .config("spark.driver.bindAddress", "127.0.0.1") \
        #             .getOrCreate()
        
        #set random state to different but fixed value per trial
        random_states = [12,56,123]
        try:
            #k_fold Cross Validation
            for j in range(2,10):
                k_fold = KFold(n_splits = j,shuffle = True, random_state = random_states[self.__trial])
                i = 1
                for split in k_fold.split(self.__train_test_df):
                    logger.info(f"Training of {j}-Fold Cross-Validation with Trainingsplit {i} started.")
                    fold_train_df = self.__train_test_df.iloc[split[0]]
                    
                    fold_test_df = self.__train_test_df.iloc[split[1]]
                    fold_test_df = fold_test_df[fold_test_df['LABEL']!= -2]
                    y_test = fold_test_df['LABEL'].to_numpy()

                    # apply labeling fdunctions on train and testset of k-fold split i
                    l_train = self.applier.apply(df=fold_train_df)
                    l_test = self.applier.apply(df=fold_test_df)

                    """Alternative approach of LF application with Snorkel SparkLFApplier for a faster application. 
                    Due to various error issues on the part of Snorkel no further utilization of this variant."""
                    # train = spark.createDataFrame(fold_train_df)
                    # l_train2 = self.applier2.apply(train.rdd)
                    # test = spark.createDataFrame(fold_test_df)
                    # l_test2 = self.applier2.apply(test.rdd)
                    
                    #analyse coverage, overlaps and conflicts of current k-fold split i
                    L_analyis = LFAnalysis(L=l_train, lfs=self.lfs)
                    self.__analysis_training_result(j,i,L_analyis.lf_summary())
                    logger.info(f"Training set coverage of {j}-Fold Cross Validation with Trainingset {i}: {100 * L_analyis.label_coverage(): 0.1f}%")

                    # optimization loop with random search, grid search and bayesian optimization
                    self.train_model(l_train,l_test,y_test,j,i)
                    logger.info(f"{j}-Fold Cross-Validation with Trainingsplit {i} were trained.")

                    #read current best result saved in temp file
                    __t_path = r'models\label\model_tuning_'+self.__lang+r'\results\temp_eval_results_'+self.__text_col+r'.feather'
                    path = str(os.path.dirname(__file__)).split("src")[0]
                    result_df = pd.read_feather(path+__t_path).sort_values(by=['accuracy'], ascending=[False])
                    current_best = result_df.to_dict('records')[0]
                    logger.info(f"Current best training: {current_best}")
                    i+=1
            
            ##save evaluation results##
            final_result_df = self.save_results()

            ##get best parameter settings and associated model##
            final_param = final_result_df.to_dict('records')[0]            
            logger.info(f"Best Model with {final_param['k-fold']}-fold Cross Validation identified. \n Maximum reached accuracy: {final_param['accuracy']}. \n Path: {final_param['model']}")

            self.__label_model = LabelModel(cardinality = 7, verbose=False)
            self.__label_model.load(final_param['model'])
        except KeyboardInterrupt:
            ##save evaluation results##
            final_result_df = self.save_results()
            
            ##get best parameter settings and associated model##
            if not final_result_df.empty:
                final_param = final_result_df.to_dict('records')[0]
                logger.info(f"Best Model with {final_param['k-fold']}-fold Cross Validation identified. \n Maximum reached accuracy: {final_param['accuracy']}. \n Path: {final_param['model']}")

                self.__label_model = LabelModel(cardinality = 7, verbose=False)
                self.__label_model.load(final_param['model'])
                ##test and save model##
                self.__test_model()
                self.__save_model()
                return
            else:
                return
    
    @classmethod
    def __analysis_training_result(self, k, i,coverage_df):
        """Analysation of applied Labeling Functions (LF) and the polarity, coverage, conflicts and overlaps (in %) of each LF on the dataset. Save coverage in dedicated result folder.
            polarity = The set of unique labels this LF outputs (excluding abstains)
            coverage = percentage of objects the LF labels
            overlap  = percentage of objects with more than one label. 
            conflict = percentage of objects with conflicting labels.
        """
        if not os.path.exists(str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\results"):
            os.makedirs(str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\results")
        logger = logging.getLogger("Labeler")
        logger.info(f"Validate metrics of {k}-fold Cross Validation with Trainingset {i}")
        
        #get current trial number
        eval_path = r'models\label\model_tuning_'+self.__lang+r'\results\eval_results_'+self.__text_col+r'.feather'
        path = str(os.path.dirname(__file__)).split("src")[0]
        
        #save coverage data next to eval data
        coverage_path = r'models\label\model_tuning_'+self.__lang+r'\results\coverage_results_'+self.__text_col+r'.feather'
        coverage_df['LF'] = coverage_df.index
        coverage_df['TRIAL'] = self.__trial
        coverage_df['k_fold'] = k
        coverage_df['k_fold_split'] = i
        coverage_df['TEXT'] = self.__text_col
        
        #save current coverage data to existing coverage data
        if os.path.exists(path+coverage_path):
            all_coverage_df = pd.read_feather(path+coverage_path)
            coverage_df_all = pd.concat([all_coverage_df,coverage_df])
            coverage_df_all.reset_index(inplace=True, drop = True)
            coverage_df_all.to_feather(path+coverage_path)
        else:
            coverage_df.reset_index(inplace=True, drop = True)
            coverage_df.to_feather(path+coverage_path)

    @classmethod
    def train_model(self,l_train,l_test,y_test,j,u):
        """Train function of model. Follows the usual procedure consisting (Hyper-)parameter Tuning.
        Selected optimizer are: Grid Search, Random Search and Bayesian Optimization. 
        """    
        _L_train_fold, _L_test_fold, _Y_test, _k, _i = l_train,l_test,y_test,j,u
        logger = logging.getLogger("Labeler")
        logger.info(f"Training and Evaluation of best Model with {_k}-fold Cross Validation with Trainingset {_i}")

        #random search
        loggerr = logging.getLogger("Labeler.randomSearch")
        try:
            self.__apply_randomSearch(_L_train_fold, _L_test_fold,_Y_test,_k,_i)
        except Exception as e:
            loggerr.warning("Error occurred: ", e)
            pass
        
        #grid search
        loggerg = logging.getLogger("Labeler.gridSearch")
        try:
            self.__apply_gridSearch(_L_train_fold, _L_test_fold,_Y_test,_k,_i)
        except Exception as e:
            loggerg.warning("Error occurred: ", e)
            pass

        #bayesian search
        loggerb = logging.getLogger("Labeler.bayesianOptim")
        try:
            self.__apply_bayesianOptimization(_L_train_fold, _L_test_fold,_Y_test,_k,_i)
        except Exception as e:
            loggerb.warning("Error occurred: ", e)
            pass

    @loggingdecorator("label.function")
    def __apply_randomSearch(self,train,test,y_test,k:int,i:int, max_evals = 40):
        """Hyperparameter Optimization techinique of Random Search Optimization.
        Save model checpoint in specific path and saves model parameter and metrics globally.

        Args:
            train (_type_): Trainset to train model with.
            test (_type_): Testset to test model with.
            y_test (_type_): True prediction values of trainset.
            k (int): Number of k-fold Cross Validation.
            i (int): Number of fold of k-fold Cross Validation.
            max_evals (int, optional): Number of optimization loops. Defaults to 40.
        """
        
        # Set up random search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\random_search_"+self.__text_col
        
        #  Choose random hyperparameters until reach max evaluations of configuration space
        eval_data_intern = []
        n_epochs = [random.randint(10,800) for _ in range(max_evals)]
        log_freq = [random.randint(10,200) for _ in range(max_evals)]
        l2 = [round(random.uniform(0.1,0.4),1) for _ in range(max_evals)]
        lr = [round(random.uniform(0.001,0.02),3)for _ in range(max_evals)]
        optimizer = [random.choice(["sgd", "adam", "adamax"])for _ in range(max_evals)]
        for l in range(max_evals):
            label_model = LabelModel(cardinality = 7, verbose=False, device = 'cuda:0')
            label_model.fit(L_train=train, n_epochs=n_epochs[l], seed=123, log_freq=log_freq[l], l2=l2[l], lr=lr[l], optimizer = optimizer[l])

            #Metrics: `accuracy`, `coverage`,`precision`, `recall`, `f1`, `f1_micro`, `f1_macro`, `fbeta`,`matthews_corrcoef`, `roc_auc`
            label_model_metrics = label_model.score(L=test, Y=y_test, metrics=['accuracy', 'coverage','precision_micro','precision_macro','f1_micro', 'f1_macro','matthews_corrcoef'],\
                                                    tie_break_policy="random")
            
            eval_data_intern.append({'Type':'RandomSearch','n_epochs':n_epochs[l],'log_freq':log_freq[l],'l2':l2[l],'lr':lr[l],'optimizer':optimizer[l],\
                                     'accuracy':label_model_metrics["accuracy" ],'PrecisionMicro':label_model_metrics['precision_micro'],'PrecisionMacro':label_model_metrics['precision_macro'],\
                                        'F1Micro':label_model_metrics['f1_micro'],'F1Macro':label_model_metrics['f1_macro'],'MCC':label_model_metrics['matthews_corrcoef'],\
                                            'Coverage':label_model_metrics['coverage'],'k-fold':k,'trainingset':i})
        
        # sort results by accuracy
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]
            
        # Set up model checkpoint folder and save model including log
        rand_name = str(time.time())[-5:] 
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)
        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'])
        label_model.save(model_folder+r"\label_model.pkl")

        # add best result to overall evaluation results
        self.__save_current_result(k,i,'RandomSearch', best_model, model_folder)


    @loggingdecorator("label.function")
    def __apply_gridSearch(self,train,test,y_test,k:int,i:int):
        """Hyperparameter Optimization techinique of Grid Search Optimization.
        Save model checpoint in specific path and saves model parameter and metrics globally.

        Args:
            train (_type_): Trainset to train model with.
            test (_type_): Testset to test model with.
            y_test (_type_): True prediction values of trainset.
            k (int): Number of k-fold Cross Validation.
            i (int): Number of fold of k-fold Cross Validation.
        """

        # Set up grid search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\grid_search_"+self.__text_col
        
        #set configuration space
        hyperparameter_space ={
            'n_epochs':[10,30,50,100],
            'log_freq':[30,90,170],
            'l2':[0.1,0.2,0.3],
            'lr':[0.00001,0.0001, 0.001],
            'optimizer':["sgd", "adam", "adamax"]
        }
        permutations = list(product(hyperparameter_space['n_epochs'],hyperparameter_space['log_freq'],hyperparameter_space['l2'],hyperparameter_space['lr'],hyperparameter_space['optimizer']))
        eval_data_intern = []
        for p in permutations:
            #logger.info(f"Selected parameter {p}")
            label_model = LabelModel(cardinality = 7, verbose=False, device = 'cuda:0')
            label_model.fit(L_train=train, n_epochs=p[0], seed=123, log_freq=p[1], l2=p[2], lr=p[3], optimizer = p[4])

            #Metrics: `accuracy`, `coverage`,`precision`, `recall`, `f1`, `f1_micro`, `f1_macro`, `fbeta`,`matthews_corrcoef`, `roc_auc`
            label_model_metrics = label_model.score(L=test, Y=y_test, metrics=['accuracy', 'coverage','precision_micro','precision_macro','f1_micro', 'f1_macro','matthews_corrcoef'],\
                                                    tie_break_policy="random")
            
            eval_data_intern.append({'Type':'GridSearch','n_epochs':p[0],'log_freq':p[1],'l2':p[2],'lr':p[3],'optimizer':p[4],\
                                     'accuracy':label_model_metrics["accuracy" ],'PrecisionMicro':label_model_metrics['precision_micro'],'PrecisionMacro':label_model_metrics['precision_macro'],\
                                        'F1Micro':label_model_metrics['f1_micro'],'F1Macro':label_model_metrics['f1_macro'],'MCC':label_model_metrics['matthews_corrcoef'],\
                                            'Coverage':label_model_metrics['coverage'],'k-fold':k,'trainingset':i})
            
        # sort results by accuracy
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]

        # Set up model checkpoint folder and save model including log
        rand_name = str(time.time())[-5:]
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)

        # add best result to overall evaluation results
        self.__save_current_result(k,i,'GridSearch', best_model, model_folder)

        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'])
        label_model.save(model_folder+r"\label_model.pkl")


    @loggingdecorator("label.function")
    def __apply_bayesianOptimization(self,train,test,y_test,k:int,i:int, max_evals = 40):
        """Hyperparameter Optimization techinique of Bayesian Optimization using bayes_opt. 
        Save model checpoint in specific path and saves model parameter and metrics globally.

        Args:
            train (_type_): Trainset to train model with.
            test (_type_): Testset to test model with.
            y_test (_type_): True prediction values of trainset.
            k (int): Number of k-fold Cross Validation.
            i (int): Number of fold of k-fold Cross Validation.
            max_evals (int, optional): Number of optimization loops. Defaults to 40.

        """
        # Set up bayesian optimization search checkpoint folder
        eval_folder = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\bayes_search_"+self.__text_col

        #configuration space
        hyperparameter_space ={
            'n_epochs':[10,150],
            'log_freq':[10,200],
            'l2':[0.1,0.5],
            'lr':[0.0001,0.01],
            'optimizer':[0,2]
        }
        eval_data_intern = []

        def model_train(n_epochs,log_freq,l2,lr, optimizer):
            """Internal bayes_opt target function.

            Args:
                n_epochs (_type_): Number of epochs to train model with.
                log_freq (_type_): Log Frequencie to train model with.
                l2 (_type_): L2-Regularization to train model with.
                lr (_type_): Learning Rate to train model with.
                optimizer (_type_): Selected Optimizer to train model with.

            Returns:
                _type_: _description_
            """
            label_model = LabelModel(cardinality = 7, verbose=False, device = 'cuda:0')
            if round(optimizer) == 0:
                optim = "sgd"
            elif round(optimizer) == 1:
                optim = "adam"
            elif round(optimizer) == 2:
                optim = "adamax"
            else:
                optim = "adam"

            label_model.fit(L_train=train, n_epochs=round(n_epochs), seed=123, log_freq=round(log_freq), l2=l2, lr=lr, optimizer = optim)

            # Metrics: `accuracy`, `coverage`,`precision`, `recall`, `f1`, `f1_micro`, `f1_macro`, `fbeta`,`matthews_corrcoef`, `roc_auc`
            # Manually added metrics in snorkel metrics.py: precision_micro  and precision_macro
            label_model_metrics = label_model.score(L=test, Y=y_test, metrics=['accuracy', 'coverage','precision_micro','precision_macro','f1_micro', 'f1_macro','matthews_corrcoef'],\
                                                    tie_break_policy="random")
            
            # Save evaluation results temporarily
            eval_data_intern.append({'Type':'BayesSearch','n_epochs':round(n_epochs),'log_freq':round(log_freq),'l2':l2,'lr':lr,'optimizer':optim,\
                                     'accuracy':label_model_metrics["accuracy" ],'PrecisionMicro':label_model_metrics['precision_micro'],'PrecisionMacro':label_model_metrics['precision_macro'],\
                                        'F1Micro':label_model_metrics['f1_micro'],'F1Macro':label_model_metrics['f1_macro'],'MCC':label_model_metrics['matthews_corrcoef'],\
                                            'Coverage':label_model_metrics['coverage'],'k-fold':k,'trainingset':i})
            return label_model_metrics["accuracy"]

        optimizer = bayes_opt.BayesianOptimization(f=model_train,pbounds =hyperparameter_space ,verbose = 1, random_state = 4)
        optimizer.maximize(init_points = 5, n_iter = max_evals)        

        # best_parameters = optimizer.max["params"]
        # highest_accuracy = optimizer.max["target"]

        # sort results by accuracy of all internal optimization loops and takes best result
        df = pd.DataFrame(eval_data_intern).sort_values(by=['accuracy'], ascending=[False])
        best_model = df.to_dict('records')[0]

        # set up model checkpoint folder to save model in
        rand_name = str(time.time())[-5:] 
        model_folder = eval_folder+r"\model_"+rand_name+'_n_epochs_'+str(best_model['n_epochs'])+'_log_freq_'+str(best_model['log_freq'])+'_l2_'+str(best_model['l2'])+'_lr_'+str(best_model['lr'])+'_optimizer_'+best_model['optimizer']+"_"+str(date.today())
        os.makedirs(model_folder)

        # add best result to overall global evaluation results 
        self.__save_current_result(k,i,'BayesSearch', best_model, model_folder)

        # save best model of all internal optimization loops to model checkpoint folder 
        label_model = LabelModel(cardinality = 7, verbose=False)
        label_model.fit(L_train=train, n_epochs=best_model['n_epochs'], seed=123, log_freq=best_model['log_freq'], l2=best_model['l2'], lr=best_model['lr'], optimizer = best_model['optimizer'])
        label_model.save(model_folder+r"\label_model.pkl")
        
    @classmethod
    def __test_model(self):
        """Validation function of model. Follows the usual procedure consisting testing of the optimized trained model with validation set. 
        If an trained model already exists on dedicated path the new trained model will be compared with existing model.
        New model will be saved and applied on whole data set if higher accuracy reached on validation set.
        """
        logger = logging.getLogger("Labeler")
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(self.__lang)+"_"+self.__text_col+r".pkl"
        
        # apply labeling functions on previously untouched validation data
        Y_val = self.__validate_df['LABEL'].to_numpy()
        L_val = self.applier.apply(df=self.__validate_df)
        preds_val_label = self.__label_model.predict(L=L_val)
        
        # apply model with best configurateion
        validate_acc = self.__label_model.score(L=L_val, Y=Y_val, tie_break_policy="random")["accuracy"]
        logger.info(f"Accuracy on Validation set: {validate_acc}")

        # load existing model for comarison reasons
        if os.path.exists(path):
            old_label_model = LabelModel(cardinality = 7, verbose=False)            
            old_label_model.load(path)
            old_model_acc = old_label_model.score(L=L_val, Y=Y_val, tie_break_policy="random")["accuracy"] 
            
            # save model if its best one in comparison to existing saved model
            if validate_acc > old_model_acc:
                self.__update_model = True
                self.__update_data = True
                logger.info(f"New model accuracy: {validate_acc}. Saved model accuracy: {old_model_acc} (Applied on Validation Set)\nModel will be overwritten.")
            else:
                self.__update_model = False
                self.__update_data = False
                logger.info(f"New model accuracy: {validate_acc}. Saved model accuracy: {old_model_acc} (Applied on Validation Set)\nModel will not be overwritten.")
        else:
            self.__update_model = True
            self.__update_data = True            
            logger.info(f"Model accuracy: {validate_acc}. No existing Model. Model will be saved.")
        
        validation_df = self.__validate_df
        validation_df['LABEL'] = preds_val_label

        # save labeled validation data for further manual check
        if self.__update_data:
            validation_df.to_feather(str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_"+self.__lang+r"\results\validation_set_"+self.__text_col+r".feather")

        before_val_shape = self.__validate_df.shape
        validation_df = validation_df[validation_df['LABEL'] != -1]
        after_val_shape = validation_df.shape
        print(f"Shape of Validationset before: {before_val_shape} and after: {after_val_shape}")
    
    @classmethod
    def __save_model(self):
        """Saving of trained Label model as pickle file with optimized parameter.
        """
        logger = logging.getLogger("Labeler")
        path = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\trained_model_"+str(self.__lang)+"_"+self.__text_col+r".pkl"

        if self.__update_model:
            self.__label_model.save(path)
            logger.info("Model saved!")
        else:
            logger.info(f"Model not saved!")
            self.__update_data = False

    @classmethod
    def __apply_model(self):
        """Apply trained model with best configuration on whole dataset and saves datasets if better than existing model.
        """
        logger = logging.getLogger("Labeler")
        if self.__update_data:
            
            logger.info(f"Language:{self.__lang}. Applying trained model with best parameters on whole data set.")

            self.L_data = self.applier.apply(df= self.__data)
            label = self.__label_model.predict(L=self.L_data)

            self.__data['LABEL'] = label
            before_data_shape = self.__data.shape
            self.__data = self.__data[self.__data['LABEL'] != -1]
            after_data_shape = self.__data.shape
            
            logger.info(f"Shape of Dataset before: {before_data_shape} and after: {after_data_shape}")
        else:
            logger.info(f"Language:{self.__lang}. NOT Applying trained model with best parameters and trainset on whole data set due to worse model accuracy.")

    @classmethod
    def __save_data(self):
        """Saves labeled data as feather into files folder.
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + self.__target_path
        logger = logging.getLogger("Labeler")
        
        self.__data.reset_index(inplace=True)
        self.__data.rename(columns = {'text':self.__text_col},inplace=True)

        if self.__update_data:
            self.__data.to_feather(path)
            logger.info("Data with applied labels saved!")
        else:
            logger.info("Data with applied labels not saved!")

    @classmethod
    def __save_current_result(self, k:int,i:int,optim:str, best_model:dict, model_folder:str):
        """Saves intermediate (best) evaluation results into a temporary file.

        Args:
            k (int): k of k-fold cross validation. Defaults to 0 as input if no k-fold cross validation is applied.
            i (int): i if split i in k-fold cross validation. Defaults to 0 as input if no k-fold cross validation is applied.
            optim (str): Type of Optimization technique. It can be differnatiated between Random Search, Hyperband or BOHB.
            best_model (dict): Best Configuration of the Optimization.
            model_folder (str): Path to trained model with best configuration.
        """
        t_path = r'models\label\model_tuning_'+self.__lang+r'\results\temp_eval_results_'+self.__text_col+r'.feather'
        path = str(os.path.dirname(__file__)).split("src")[0]
        evaluation_data = [{'Type':optim,'n_epochs':best_model['n_epochs'],'log_freq':best_model['log_freq'],'l2':best_model['l2'],'lr':best_model['lr'],\
                                     'optimizer':best_model['optimizer'],\
                                        'accuracy':best_model["accuracy" ],'PrecisionMicro':best_model['PrecisionMicro'],'PrecisionMacro':best_model['PrecisionMacro'],\
                                            'F1Micro':best_model['F1Micro'],'F1Macro':best_model['F1Macro'],'MCC':best_model['MCC'],'Coverage':best_model['Coverage'],\
                                                'k-fold':k,'trainingset':i, 'model':model_folder+r"\label_model.pkl", 'Trial': self.__trial, 'TEXT': self.__text_col}]
        df_new = pd.DataFrame(evaluation_data)
        
        if os.path.exists(path+t_path):
            df_all = pd.read_feather(path+t_path)
            df_all_new = pd.concat([df_all,df_new])
        else:
            df_all_new = df_new
        
        df_all_new.reset_index(inplace = True, drop = True) 
        df_all_new.to_feather(path+t_path)

    @classmethod
    def save_results(self):
        """Saves evaluation results to dedicated result folder.
        """
        _path = str(os.path.dirname(__file__)).split("src")[0]
        _temp_t_path = r'models\label\model_tuning_'+self.__lang+r'\results\temp_eval_results_'+self.__text_col+r'.feather'
        t_path = r'models\label\model_tuning_'+self.__lang+r'\results\eval_results_'+self.__text_col+r'.feather'

        #get temp eval data
        temp_df = pd.read_feather(_path+_temp_t_path)
        
        ##check if target_path already exists
        if os.path.exists(_path+t_path):
            df_all = pd.read_feather(_path+t_path)
            df_all_new = pd.concat([df_all,temp_df])
        else:
            df_all_new = temp_df

        #save temp to existing eval data and remove temp file           
        df_all_new.reset_index(inplace = True, drop = True)
        df_all_new.to_feather(_path+t_path)
        os.remove(_path+_temp_t_path)
        
        temp_df.sort_values(by=['accuracy'], ascending=[False], inplace=True)
        return temp_df

    @classmethod
    def __get_trial(self) -> int:
        """Gets current trial of training of a Label Model.

        Returns:
            int: Returns current trial number as integer.
        """
        t_path = r'models\label\model_tuning_'+self.__lang+r'\results\eval_results_'+self.__text_col+r'.feather'
        path = str(os.path.dirname(__file__)).split("src")[0]
        trial = 0
        ##check if target_path already exists
        if os.path.exists(path+t_path):
            df_all = pd.read_feather(path+t_path)
            #get current number of trial
            trial = df_all[['Trial']].sort_values(by=['Trial'], ascending=[False]).to_dict('records')[0]['Trial'] + 1

        return trial
    
    def run(self):
        """Main function. Combines the training of the model with the different hyperparameter optimization techniques, 
            the validation of the best model, applies it on the data and stores everything including the evaluation results.
        """
        #Start Label Model training and application
        logger = logging.getLogger("Labeler")
        logger.info(f"############################################################################ Run {self.__trial} - {self.__text_col} ########################################################################################")
        logger.info("Automated Labeling started with Language {l}, Text-Column: {t_col} and data file {path} (source) created. Target file is {tpath}".format(l = lang, path = self.source_path,\
                                                                                                                                                               tpath = self.__target_path,\
                                                                                                                                                                t_col = self.__text_col))
        
        #apply labeling functions on data splits, train model and optimize parameter
        self.__train_labeling_functions()
        #test model with best configuration on validation set
        self.__test_model()
        #save best model
        self.__save_model()
        #apply best model on whole dataset 
        self.__apply_model()
        #save dataset with assigned labels
        self.__save_data()

for lang in ['en']:
    for i in range(3):
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_TOPIC'+".feather",'TOPIC', True).run()
    for i in range(3):
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_URL_TEXT'+".feather",'URL_TEXT',True).run()
   
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

