import pandas as pd
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

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
    # for lang in ['de','en']:
    #     for text_col in ['TOPIC', 'URL_TEXT']:
    coverage = pd.read_feather(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'backup\models\label\model_tuning_de\results\coverage_results_TOPIC.feather')
    coverage.sort_values(by = ['TRIAL','k_fold','k_fold_split','Coverage','Overlaps','Conflicts'], ascending = [False,False,False,False,True,True], inplace = True)
    coverage.drop_duplicates(subset=['LF'], inplace=True, keep='first')
    coverage['Polarity'] = coverage['Polarity'].replace({0:'AUTONOMOUS', 1:'CONNECTIVITY',2:'DIGITALISATION',3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"})
    coverage["LF"] = coverage['Polarity'] +"_"+coverage["LF"].str.split("_", n = 1, expand = True)[0] 
    plot_coverage(coverage, lang='de', text_col='TOPIC')


    eval = pd.read_feather(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'backup\models\label\model_tuning_de\results\eval_results_TOPIC.feather')
    eval.sort_values(by = ['Trial','k-fold','trainingset','accuracy','MCC'], ascending = [False,False,False,False,False], inplace = True)
    eval.drop_duplicates(subset=['k-fold'], inplace=True, keep='first')
    plot_eval(eval, lang='de', text_col='TOPIC')
    # coverage_de_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_de\results\coverage_results_TOPIC.feather')
    # eval_de_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_de\results\eval_results_TOPIC.feather')
    # coverage_en_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_en\results\coverage_results_TOPIC.feather')
    # eval_en_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_en\results\eval_results.feather')
    
    #return coverage_de_TOPIC,eval_de_TOPIC

def laod_eval_data_classification():
    pass


def plot_coverage(df, lang, text_col):
    ind = np.arange(len(df['LF'].tolist()))
    width = 0.25

    fig, ax = plt.subplots()

    ax.barh(ind, df.Coverage, width, color='#478c2e', label='Coverage')
    ax.barh(ind + width, df.Overlaps, width, color='#2e308c', label='Overlaps')
    ax.barh(ind + 2*width, df.Conflicts, width, color='#8c2e2e', label='Conflicts')

    ax.set(yticks=ind + width, yticklabels=df.LF, ylim=[2*width - 1, len(df)])
    ax.legend()
    # ax.set_title("TITLE")

    ax.margins(x = 0.5,y = 0)
    plt.subplots_adjust(left=0.33, bottom = 0.05, top = 0.99, right = 0.99)
    
    plt.savefig(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'\backup\images\fig2.png')#pdf
    plt.close()

def plot_eval(df, lang, text_col):
    barWidth = 0.25
    label = df['k-fold'].tolist()

    r1 = np.arange(len(df['accuracy'].tolist()))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, df['accuracy'].tolist(), color='#478c2e', width=barWidth, edgecolor='white', label='Accuracy')
    plt.bar(r2, df['MCC'].tolist(), color='#2e308c', width=barWidth, edgecolor='white', label='MCC')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('K-Fold', fontweight='bold')
    plt.xticks([r + barWidth-0.125 for r in range(len(df['accuracy'].tolist()))], label)
    
    # Create legend & Show graphic
    plt.legend()
    
    plt.savefig(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'\backup\images\fig3.png')#pdf
    plt.close()


def plot_group_barchart(df,x_,y_,group):
    fig = px.bar(df, x=x_, y=y_,
             color=group, barmode='group',
             height=400)
    fig.write_image(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'\backup\images\fig1.png')#pdf
    # fig.show()
def plot_histogramm(df,x,color):
    fig = px.histogram(data_frame = df,x = x,color = color)
    fig.write_image(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'\backup\images\fig2.png')#pdf


# raw_en,raw_de = load_raw_data()
# clean_en, clean_de = load_clean_data()
# labeled_en_TOPIC,labeled_de_TOPIC,labeled_en_URL_TEXT,labeled_de_URL_TEXT = load_labeled_data()
load_eval_data_automated_label()
# print(raw_en.shape)
# print(clean_en.shape)
# print(labeled_en_TOPIC.shape)
# print(labeled_en_URL_TEXT.shape)