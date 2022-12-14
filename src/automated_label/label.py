import pandas as pd
import os
import numpy as np
import pickle
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel

ABSTAIN = 1
AUTONOMOUS = 2
ELECTRIFICATION = 3
CONNECTIVITY = 4
SHARED = 5
SUSTAINABILITY = 6
DIGITALISATION = 7
INDIVIDUALISATION = 8

path = str(os.path.dirname(__file__)).split("src")[0] + r"files/CLASS_keywords.json"
DATA = pd.read_json(path)

#X DEFINE NUMBER LABELS 1-7###
#X DEFINE CLUSTERING MODEL (SIMPLE)###
#X SELECT ALL KEYWORDS BASED ON SUFFICENT SOURCES###
###DEFINE 2 HEURISTICS PER LABEL###
###DEFINE SIMPLE "BLACKLIST" PER LABEL###

@labeling_function()
def predict_cluster(x):
    value = ABSTAIN
    kmeans = pickle.load(open("kmeans.pkl", 'rb'))           
    cluster = kmeans.predict(x)
    if cluster == 1:
        value = AUTONOMOUS
        # upload cluster dictionary
        #.
        #.
        #.

    return value

@labeling_function()
def check_autonomous(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'AUTONOMOUS']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = AUTONOMOUS
    return value

@labeling_function()
def check_electrification(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'ELECTRIFICATION']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = ELECTRIFICATION       
    return value
    
@labeling_function()
def check_digitalisation(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'DIGITALISATION']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = DIGITALISATION
    return value

@labeling_function()
def check_connectivity(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'CONNECTIVITY']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = CONNECTIVITY
    return value

@labeling_function()
def check_sustainability(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'SUSTAINABILITY']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = SUSTAINABILITY
    return value

@labeling_function()
def check_individualisaton(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'INDIVIDUALISATION']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = INDIVIDUALISATION
    return value

@labeling_function()
def check_shared(x):
    value = ABSTAIN
    keywords = DATA[DATA["CLASS"] == 'SHARED']['KEYWORDS'].tolist()
    if any(key.lower() in x.text.lower() for key in keywords):
        value = SHARED
    return value

class Labeler:
    def __init__(self):
        self.data = self.load_data()
        self.train_df = self.data
        # self.validate_df = self.data
        # self.validate_labels = self.data['LABEL']
        self.lfs = [check_autonomous,check_connectivity,check_digitalisation,check_electrification,check_individualisaton,check_shared,check_sustainability]
        self.L_train = None
        self.label_model = None

    def load_data(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_cleaned.csv"
        df = pd.read_csv(df_path, header = 0, delimiter=",")
        return df.replace(np.nan, "",regex = False)

    def save_data(self):
        self.data.to_csv(str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_labeled.csv", index = False)

    def apply_labeling_functions(self):
        self.data = self.data.rename(columns = {'URL-TEXT' : 'text'})

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
        self.data.loc[(self.data['LABEL']==1), 'LABEL'] ='ABSTAIN'
        self.data.loc[(self.data['LABEL']==2), 'LABEL'] ='GENERAL'
        self.data.loc[(self.data['LABEL']==3), 'LABEL'] ='SPECIFIC'
        self.data.loc[(self.data['LABEL']==4), 'LABEL'] ='ELECTRIC'

    def analysis_result_model(self):
        print('validate metrics')
        #print(self.label_model.score(L_validate, Y=validate_labels,metrics=["f1","accuracy",'precision','recall']))
    
    def run(self):
        self.apply_labeling_functions()
        self.apply_labeling_model()
        #self.assign_labels_final()
        self.save_data()

if __name__ == "__main__":
    l = Labeler()
    l.run()
    print(l.data['LABEL'].tolist())
