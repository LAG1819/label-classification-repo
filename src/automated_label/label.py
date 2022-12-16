import pandas as pd
import os
import numpy as np
import pickle
import re
import operator
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel

ABSTAIN = 1
AUTONOMOUS = 2
CONNECTIVITY = 3
DIGITALISATION = 4
ELECTRIFICATION = 5
INDIVIDUALISATION = 6
SHARED = 7
SUSTAINABILITY = 8

path = str(os.path.dirname(__file__)).split("src")[0] + r"files/CLASS_keywords.json"
DATA = pd.read_json(path)

#X DEFINE NUMBER LABELS 1-7###
#X DEFINE CLUSTERING MODEL (SIMPLE)###
#X SELECT ALL KEYWORDS BASED ON SUFFICENT SOURCES###
###DEFINE 2 HEURISTICS PER LABEL###

@labeling_function()
def predict_cluster(x):
    #load trained kMeans, fitted vectorizer and cluster dictionary
    cluster_names = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\kMeans_cluster.feather").to_dict('records')[0]
    kmeans = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans.pkl", 'rb')) 
    kmeans_vectorizer = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/kmeans_vectorizer.pkl", 'rb')) 
    
    text = kmeans_vectorizer.transform([str(x)]).toarray()
    cluster = kmeans.predict(text)[0]+2

    value = ABSTAIN
    list_of_values = {ABSTAIN:"ABSTAIN",AUTONOMOUS:"AUTONOMOUS",ELECTRIFICATION:"ELECTRIFICATION",CONNECTIVITY:"CONNECTIVITY",SHARED:"SHARED",\
            SUSTAINABILITY:"SUSTAINABILITY",DIGITALISATION:"DIGITALISATION",INDIVIDUALISATION:"INDIVIDUALISATION"}
    try:
        number = cluster_names[str(cluster)]
        value = list_of_values[number]
    except IndexError:
        value = ABSTAIN
    except KeyError:
        value = ABSTAIN

    return value

@labeling_function()
def check_keywords(x):
    value = ABSTAIN
    dic = {}
    keywords = DATA[DATA["CLASS"] == 'AUTONOMOUS']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[AUTONOMOUS] = counter

    keywords = DATA[DATA["CLASS"] == 'ELECTRIFICATION']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[ELECTRIFICATION] = counter
    
    keywords = DATA[DATA["CLASS"] == 'DIGITALISATION']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[DIGITALISATION] = counter
    
    keywords = DATA[DATA["CLASS"] == 'CONNECTIVITY']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[CONNECTIVITY] = counter

    keywords = DATA[DATA["CLASS"] == 'SUSTAINABILITY']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[SUSTAINABILITY] = counter

    keywords = DATA[DATA["CLASS"] == 'INDIVIDUALISATION']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[INDIVIDUALISATION] = counter 
    
    keywords = DATA[DATA["CLASS"] == 'SHARED']['KEYWORDS'].tolist()[0]
    counter = 0
    for key in keywords:
        if re.search(key.lower(), x.text.lower()):
            counter +=1
    dic[SHARED] = counter

    return max(dic, key=dic.get)

@labeling_function()
def check_random(x):
    return AUTONOMOUS

# @labeling_function()
# def check_autonomous(x):
#     value = ABSTAIN
#     keywords = DATA[DATA["CLASS"] == 'AUTONOMOUS']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = AUTONOMOUS
#     return value

# @labeling_function()
# def check_electrification(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'ELECTRIFICATION']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = ELECTRIFICATION       
#     return value
    
# @labeling_function()
# def check_digitalisation(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'DIGITALISATION']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = DIGITALISATION
#     return value

# @labeling_function()
# def check_connectivity(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'CONNECTIVITY']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = CONNECTIVITY
#     return value

# @labeling_function()
# def check_sustainability(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'SUSTAINABILITY']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = SUSTAINABILITY
#     return value

# @labeling_function()
# def check_individualisaton(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'INDIVIDUALISATION']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = INDIVIDUALISATION
#     return value

# @labeling_function()
# def check_shared(x):
#     value = ABSTAIN
    
#     keywords = DATA[DATA["CLASS"] == 'SHARED']['KEYWORDS'].tolist()[0]
#     if any(re.search(key.lower(), x.text.lower()) for key in keywords):
#         value = SHARED
#     return value

class Labeler:
    def __init__(self,s_path:str, t_path:str):
        # self.train_df = self.data
        # self.validate_df = self.data
        # self.validate_labels = self.data['LABEL']
        self.lfs = [predict_cluster,check_keywords, check_random]#[check_autonomous,check_connectivity,check_digitalisation,check_electrification,check_individualisaton,check_shared,check_sustainability]
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = 'URL_TEXT'
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: Returns pandas DataFrame containing the cleaned texts and its topics
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path)
        df = df.rename(columns = {self.text_col : 'text'})
        return df.replace(np.nan, "",regex = False)

    def save_data(self):
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

    def apply_labeling_functions(self):
        applier = PandasLFApplier(lfs=self.lfs)
        self.L_train = applier.apply(df=self.data)
        #print(self.L_train)

        # coverage_checking_out, coverage_check = (self.L_train != ABSTAIN).mean(axis=0)
        # print(f"checking specific coverage: {coverage_checking_out * 100:.1f}%")
        # print(f"checking general coverage: {coverage_check * 100:.1f}%")

        L_analyis = LFAnalysis(L=self.L_train, lfs=self.lfs)
        print(L_analyis.lf_summary())

    def apply_labeling_model(self):
        self.label_model = LabelModel(cardinality = 9, verbose=False)
        self.label_model.fit(L_train=self.L_train, n_epochs=1000, seed=100)
        preds_train_label = self.label_model.predict(L=self.L_train)
        #print(preds_train_label)
        
        self.data['LABEL'] = preds_train_label
        # preds_valid_label = label_model.predict(L=L_validate)
    
    def assign_labels_final(self):
        list_of_values = {"ABSTAIN":ABSTAIN,"AUTONOMOUS":AUTONOMOUS,"ELECTRIFICATION":ELECTRIFICATION,"CONNECTIVITY":CONNECTIVITY,"SHARED":SHARED,\
            "SUSTAINABILITY":SUSTAINABILITY,"DIGITALISATION":DIGITALISATION,"INDIVIDUALISATION":INDIVIDUALISATION}
        for v in list_of_values.keys:
            self.data.loc[(self.data['LABEL']==list_of_values[v]), 'LABEL'] = v

    def analysis_result_model(self):
        print('validate metrics')
        coverage_predict_cluster,coverage_check_keywords, coverage_check_random = (self.L_train != ABSTAIN).mean(axis=0)
        print(f"coverage_predict_cluster: {coverage_predict_cluster * 100:.1f}%")
        print(f"coverage_check_keywords: {coverage_check_keywords * 100:.1f}%")
        #print(self.label_model.score(L_validate, Y=validate_labels,metrics=["f1","accuracy",'precision','recall']))
    
    def run(self):
        self.apply_labeling_functions()
        self.apply_labeling_model()
        #self.assign_labels_final()
        self.analysis_result_model()
        #self.save_data()

if __name__ == "__main__":
    l = Labeler(r"files\topiced_texts.feather",r"files\labeled_texts.feather")
    l.run()
    #print(l.data['LABEL'].tolist())

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
    