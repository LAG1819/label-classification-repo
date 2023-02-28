import pandas as pd
import os
import numpy as np

def load_raw_data():
    raw_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_en.feather")
    raw_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_de.feather")
    return raw_en,raw_de

def load_clean_data():
    clean_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_en.feather")
    clean_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_de.feather")
    return clean_en,clean_de

def load_labeled_data():
    labeled_en_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_en_TOPIC.feather")
    labeled_de_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_de_TOPIC.feather")

    labeled_en_URL_TEXT = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_en_URL_TEXT.feather")
    labeled_de_URL_TEXT = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_de_URL_TEXT.feather")
    return labeled_en_TOPIC,labeled_de_TOPIC,labeled_en_URL_TEXT,labeled_de_URL_TEXT

def load_eval_data_automated_label_old():
    file_en = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_en\automated_labeling_en.log"
    file_de = str(os.path.dirname(__file__)).split("src")[0] + r"models\label\model_tuning_de\automated_labeling_de.log"
    data = []
    with open(file_en) as f:
        for i, line in enumerate(f):
            # if "Automated Labeling started with Language" in line:
            #     data.append(line)
            if i >=1263 and i <=1292:
                dataset = [1] + list(filter(None, line.split(" "))) 
                data.append(dataset)
            if i >=3828 and i <=3857: 
                dataset = [2] + list(filter(None, line.split(" "))) 
                data.append(dataset)
            if i >=5877 and i <=5906:
                dataset = [3] + list(filter(None, line.split(" "))) 
                data.append(dataset)
            if i >=7504 and i <=7533:
                dataset = [4] + list(filter(None, line.split(" "))) 
                data.append(dataset)
    data = [d for d in data if not ("Type" in d or "Model" in d)]
    df = pd.DataFrame(data)
    df = df[[0,2,3,4,5,6,7,8,9,10]]
    df = df.rename(columns={0:"Trial",2: "Type", 3:"n_epochs",4:"log_freq",5:"l2",6:"lr",7:"optimizer",8:"accuracy",9:"k-fold",10:"trainingset"})
    df['LANG'] = 'EN' 
    print(df)

def load_eval_data_automated_label():
    coverage_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_de\results\coverage_results.feather')
    eval_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_de\results\eval_results.feather')
    eval_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_en\results\eval_results.feather')
    return eval_de,eval_en

def laod_eval_data_classification():
    pass

# raw_en,raw_de = load_raw_data()
# clean_en, clean_de = load_clean_data()
# labeled_en_TOPIC,labeled_de_TOPIC,labeled_en_URL_TEXT,labeled_de_URL_TEXT = load_labeled_data()
load_eval_data_automated_label()
# print(raw_en.shape)
# print(clean_en.shape)
# print(labeled_en_TOPIC.shape)
# print(labeled_en_URL_TEXT.shape)