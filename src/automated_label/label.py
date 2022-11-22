import pandas as pd
import os
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.label_model import LabelModel

ABSTAIN = -1
GENERAL = 0
SPECIFIC = 1
ELECTRIC = 2

@labeling_function()
def check_general(x):
    return GENERAL if "startseite" in x.text.lower() else ABSTAIN
@labeling_function()
def check_specific(x):
    return SPECIFIC if ("romeo" or "audi" or "bentley")  in x.text.lower()  else ABSTAIN

@labeling_function()
def check_electric(x):
    return SPECIFIC if ("e-fahrzeug" or "elektro" or "strom")  in x.text.lower()  else ABSTAIN

class Labeler:
    def __init__(self):
        self.data = self.load_data()
        self.train_df = self.data
        # self.validate_df = self.data
        # self.validate_labels = self.data['LABEL']
        self.lfs = [check_general,check_specific,check_electric]
        self.L_train = None
        self.label_model = None

    def load_data(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_cleaned.csv"
        return pd.read_csv(df_path, header = 0, delimiter=",")

    def save_data(self):
        self.data.to_csv(str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_labeled.csv", index = False)

    def apply_labeling_functions(self):
        self.data = self.data.rename(columns = {'TEXT' : 'text'})

        applier = PandasLFApplier(lfs=self.lfs)
        self.L_train = applier.apply(df=self.data)
        print(self.L_train)

        # coverage_checking_out, coverage_check = (self.L_train != ABSTAIN).mean(axis=0)
        # print(f"checking specific coverage: {coverage_checking_out * 100:.1f}%")
        # print(f"checking general coverage: {coverage_check * 100:.1f}%")

        # L_analyis = LFAnalysis(L=L_train, lfs=self.lfs)
        # print(L_analyis.lf_summary())

    def apply_labeling_model(self):
        self.label_model = LabelModel(verbose=False)
        self.label_model.fit(L_train=self.L_train, n_epochs=1000, seed=100)
        preds_train_label = self.label_model.predict(L=self.L_train)
        print(preds_train_label)
        
        self.data['LABEL'] = preds_train_label
        # preds_valid_label = label_model.predict(L=L_validate)
    
    def assign_labels_final(self):
        self.data.loc[(self.data['LABEL']==-1), 'LABEL'] ='ABSTAIN'
        self.data.loc[(self.data['LABEL']==0), 'LABEL'] ='GENERAL'
        self.data.loc[(self.data['LABEL']==1), 'LABEL'] ='SPECIFIC'
        self.data.loc[(self.data['LABEL']==2), 'LABEL'] ='ELECTRIC'

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
    print(l.data['LABEL'])
