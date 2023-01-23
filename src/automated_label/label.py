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

ABSTAIN = 1
AUTONOMOUS = 2
CONNECTIVITY = 3
DIGITALISATION = 4
ELECTRIFICATION = 5
INDIVIDUALISATION = 6
SHARED = 7
SUSTAINABILITY = 8

#X DEFINE NUMBER LABELS 1-7###
#X DEFINE CLUSTERING MODEL (SIMPLE)###
#X SELECT ALL KEYWORDS BASED ON SUFFICENT SOURCES###
###DEFINE 2 HEURISTICS PER LABEL###

@labeling_function()
def check_brand(x):
    brandname_list=[]
    if any(brandname in x for brandname in brandname_list):
        return AUTONOMOUS
    else:
        return ABSTAIN


@labeling_function()
def heuristic_autonomous(x):
    autonom_list = ["kognitiv", "künstlich", "software"]
    if any(ele in x for ele in autonom_list):
        return AUTONOMOUS
    else:
        return ABSTAIN

@labeling_function()
def heuristic_digitalisation(x):
    digitalisation_list = ["forschung","research","forschen", "lidar", "tech", "innovation"]
    if any(ele in x for ele in digitalisation_list):
        return DIGITALISATION
    else:
        return ABSTAIN

@labeling_function()
def heuristic_sustainability(x):
    sustain_list = ["grün","green", "umwelt","environment", "klimaschutz","climate", "energiewende","energy transition", "wasserstoff","hydrogen",\
        "bodenschutz","soil protection", "energiesicherheit","energy security","energy safety", "wärmewende", "energie","energy", "supply", "rohstoff"]
    if any(ele in x for ele in sustain_list):
        return SUSTAINABILITY
    else:
        return ABSTAIN

@labeling_function()
def heuristic_urbanisation(x):
    sustain_list = ["tageszulassung","day licence","finanzierung","financing","funding", "leasing", "uber", "share"]
    if any(ele in x for ele in sustain_list):
        return SHARED
    else:
        return ABSTAIN

@labeling_function()
def heuristic_electrification(x):
    electric_list = ["co2", "wallbox"]
    if any(ele in x for ele in electric_list):
        return ELECTRIFICATION
    else:
        return ABSTAIN

@labeling_function()
def heuristic_individualisation(x):
    indiv_list = []
    if any(ele in x for ele in indiv_list):
        return INDIVIDUALISATION
    else:
        return ABSTAIN

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
        name=f"keyword_{keywords[0]}",
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
    def __init__(self,lang:str,s_path:str, t_path:str):
        """Initialise a Label object that can label topics of cleaned texts.

        Args:
            lang (str):
            s_path (str): source path to file containing raw texts to clean
            t_path (str): target path to save file with cleaned texts
        """
        if lang == 'de':
            self.lfs = [autonomous_cluster_d, electrification_cluster_d,digitalisation_cluster_d,connectivity_cluster_d, sustainability_cluster_d,individualisation_cluster_d, shared_cluster_d,\
                autonomous_keywords,electrification_keywords,digitalisation_keywords,connectivity_keywords,sustainability_keywords,individualisation_keywords,shared_keywords]
        if lang == 'en':
            self.lfs = [autonomous_cluster_e, electrification_cluster_e, digitalisation_cluster_e, connectivity_cluster_e,sustainability_cluster_e,individualisation_cluster_e,shared_cluster_e,\
                autonomous_keywords,electrification_keywords,digitalisation_keywords,connectivity_keywords,sustainability_keywords,individualisation_keywords,shared_keywords]
        
        # self.lfs = [autonomous_cluster, autonomous_keywords,electrification_keywords]
        self.L_train = None
        self.label_model = None
        self.source_path = s_path
        self.target_path = t_path
        self.text_col = 'TOPIC'#'URL_TEXT'
        self.data = self.load_data()
        self.train_df, self.validate_df, self.test_df = self.generate_trainTestdata(lang)

    def load_data(self) -> pd.DataFrame:
        """_summary_

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
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

    def apply_labeling_functions(self):
        applier = PandasLFApplier(lfs=self.lfs)
        self.L_train = applier.apply(df=self.train_df)
        #self.L_val = applier.apply(df=self.validate_df)
        self.L_test = applier.apply(df=self.test_df)

        L_analyis = LFAnalysis(L=self.L_train, lfs=self.lfs)
        print(f"Training set coverage: {100 * L_analyis.label_coverage(): 0.1f}%")
        print(L_analyis.lf_summary())

    def analysis_training_result(self):
        print('validate metrics')
        autonomous_cl, electrification_cl,digitalisation_cl,connectivity_cl, sustainability_cl,individualisation_cl, shared_cl,\
                autonomous_k,electrification_k,digitalisation_k,connectivity_k,sustainability_k,individualisation_k,shared_k = (self.L_train != ABSTAIN).mean(axis=0)

        print(f"coverage_cluster_autonomous: {autonomous_cl * 100:.1f}%")
        print(f"coverage_cluster_electrification: {electrification_cl * 100:.1f}%")
        print(f"coverage_cluster_digitalisation: {digitalisation_cl * 100:.1f}%")
        print(f"coverage_cluster_connectivity: {connectivity_cl * 100:.1f}%")
        print(f"coverage_cluster_sustainability: {sustainability_cl * 100:.1f}%")
        print(f"coverage_cluster_individualisation: {individualisation_cl * 100:.1f}%")
        print(f"coverage_cluster_shared: {shared_cl * 100:.1f}%")
        
        print(f"coverage_keyword_autonomous: {autonomous_k * 100:.1f}%")
        print(f"coverage_keyword_digitalisation: {digitalisation_k * 100:.1f}%")
        print(f"coverage_keyword_electrification: {electrification_k * 100:.1f}%")
        print(f"coverage_keyword_sustainability: {sustainability_k * 100:.1f}%")
        print(f"coverage_keyword_connectivity: {connectivity_k * 100:.1f}%")
        print(f"coverage_keyword_individualisation: {individualisation_k * 100:.1f}%")
        print(f"coverage_keyword_shared: {shared_k * 100:.1f}%")
        
        #print(self.label_model.score(L_validate, Y=validate_labels,metrics=["f1","accuracy",'precision','recall']))

    def test_labeling_model(self):
        self.label_model = LabelModel(cardinality = 9, verbose=False)
        self.label_model.fit(L_train=self.L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)

        self.majority_model = MajorityLabelVoter(cardinality=9)
        self.majority_model.predict(L=self.L_train)
        
        Y_test = self.test_df['LABEL'].to_numpy()
        majority_acc = self.majority_model.score(L=self.L_test, Y=Y_test, tie_break_policy="random")[
            "accuracy"
        ]
        print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
        label_model_acc = self.label_model.score(L=self.L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
        ]
        print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
        # , 
        # "f1", 
        # "precision", 
        # "recall"

    def eval_labeling_model(self):
       pass

    def apply_trained_model(self):
        preds_test_label = self.label_model.predict(L=self.L_test)
        self.test_df['LABEL'] = preds_test_label
        print(self.test_df.shape)
        self.test_df = self.test_df[self.test_df['LABEL'] != 1]
        print(self.test_df.shape)
        
        preds_val_label = self.label_model.predict(L=self.L_val)
        self.validate_df['LABEL'] = preds_val_label
        print(self.validate_df.shape)
        self.validate_df = self.validate_df[self.validate_df['LABEL'] != 1]
        print(self.validate_df.shape)    

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
        self.test_labeling_model()
        #self.eval_labeling_model()
        # self.apply_labeling_model()
        #self.assign_labels_final()
        self.show_samples_per_class()
        #self.save_data()

if __name__ == "__main__":
    l = Labeler('de',r"files\topiced_texts.feather",r"files\labeled_texts.feather")
    l.run()

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
    