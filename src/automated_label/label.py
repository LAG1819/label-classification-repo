import pandas as pd
import os
import numpy as np
import pickle
import re
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
    #load trained kMeans and cluster dictionary
    cluster_names = pd.read_feather(r"files\kMeans_cluster.feather").to_dict('records')[0]
    kmeans = pickle.load(open("kmeans.pkl", 'rb')) 
    value = ABSTAIN

    cluster = kmeans.predict(x)
    list_of_values = [ABSTAIN,AUTONOMOUS,ELECTRIFICATION,CONNECTIVITY,SHARED,SUSTAINABILITY,DIGITALISATION,INDIVIDUALISATION]
    value = cluster_names[cluster]
    for v in list_of_values:
        if str(value) == str(v):
            value = v
    return value

@labeling_function()
def check_autonomous(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'AUTONOMOUS']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = AUTONOMOUS
    return value

@labeling_function()
def check_electrification(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'ELECTRIFICATION']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = ELECTRIFICATION       
    return value
    
@labeling_function()
def check_digitalisation(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'DIGITALISATION']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = DIGITALISATION
    return value

@labeling_function()
def check_connectivity(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'CONNECTIVITY']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = CONNECTIVITY
    return value

@labeling_function()
def check_sustainability(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'SUSTAINABILITY']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = SUSTAINABILITY
    return value

@labeling_function()
def check_individualisaton(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'INDIVIDUALISATION']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = INDIVIDUALISATION
    return value

@labeling_function()
def check_shared(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'SHARED']['KEYWORDS'].tolist()
    if any(re.search(key.lower(), x.text.lower()) for key in keywords):
        value = SHARED
    return value

class Labeler:
    def __init__(self,s_path:str, t_path:str):
        self.data = self.load_data()
        self.train_df = self.data
        # self.validate_df = self.data
        # self.validate_labels = self.data['LABEL']
        self.lfs = [predict_cluster,check_autonomous,check_connectivity,check_digitalisation,check_electrification,check_individualisaton,check_shared,check_sustainability]
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = 'URL_TEXT'

    def load_data(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df = pd.read_feather(df_path, header = 0, delimiter=",")
        return df.replace(np.nan, "",regex = False)

    def save_data(self):
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

    def apply_labeling_functions(self):
        self.data = self.data.rename(columns = {self.text_col : 'text'})

        applier = PandasLFApplier(lfs=self.lfs)
        self.L_train = applier.apply(df=self.data)
        #print(self.L_train)

        # coverage_checking_out, coverage_check = (self.L_train != ABSTAIN).mean(axis=0)
        # print(f"checking specific coverage: {coverage_checking_out * 100:.1f}%")
        # print(f"checking general coverage: {coverage_check * 100:.1f}%")

        # L_analyis = LFAnalysis(L=L_train, lfs=self.lfs)
        # print(L_analyis.lf_summary())

    def apply_labeling_model(self):
        self.label_model = LabelModel(cardinality = 5, verbose=False)
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
        #print(self.label_model.score(L_validate, Y=validate_labels,metrics=["f1","accuracy",'precision','recall']))
    
    def run(self):
        self.apply_labeling_functions()
        self.apply_labeling_model()
        #self.assign_labels_final()
        self.save_data()

if __name__ == "__main__":
    l = Labeler(r"files\topiced_texts.feather",r"files\labeled_texts.feather" )
    l.run()
    print(l.data['LABEL'].tolist())
