import pandas as pd
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns

def load_raw_data():
    raw_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_en.feather")
    raw_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_de.feather")
    print(raw_en.shape)
    print(raw_de.shape)

def load_clean_data():
    clean_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_en.feather")
    clean_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_de.feather")
    print(clean_en.shape)
    print(clean_de.shape)

def load_labeled_data():
    labeled_en_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_en_TOPIC.feather")
    labeled_de_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_de_TOPIC.feather")

    labeled_en_URL_TEXT = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_en_URL_TEXT.feather")
    labeled_de_URL_TEXT = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify\labeled_texts_de_URL_TEXT.feather")

    print(labeled_en_TOPIC.shape)
    print(labeled_en_URL_TEXT.shape)
    print(labeled_de_TOPIC.shape)
    print(labeled_de_URL_TEXT.shape)

def load_eval_data_automated_label():
    for lang in ['de','en']:
        for text_col in ['TOPIC', 'URL_TEXT']:
            # coverage = pd.read_feather(str(os.path.dirname(__file__)).split("ml-classification-repo")[0]+r'backup\models\label\model_tuning_'+lang+r'\results\coverage_results_'+text_col+r'.feather')
            coverage = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_'+lang+r'\results\coverage_results_'+text_col+r'.feather')
            coverage.sort_values(by = ['TRIAL','k_fold','k_fold_split','Overlaps','Conflicts','Coverage',], ascending = [False, False, False,True,True,False], inplace = True)
            coverage.drop_duplicates(subset=['LF'], inplace=True, keep='first')
            coverage['Polarity'] = coverage['Polarity'].replace({0:'AUTONOMOUS', 1:'CONNECTIVITY',2:'DIGITALISATION',3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"})
            coverage["LF"] = coverage['Polarity'] +"_"+coverage["LF"].str.split("_", n = 1, expand = True)[0] 
            plot_coverage(coverage, lang=lang, text_col=text_col)


            eval = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label\model_tuning_'+lang+r'\results\eval_results_'+text_col+r'.feather')
            
            eval2 = eval.sort_values(by = ['accuracy','MCC'], ascending = [False,False])
            eval2.drop_duplicates(subset=['Type'], inplace=True, keep='first')
            eval2 = eval2[['Type',"accuracy", "PrecisionMicro", "PrecisionMacro","F1Micro","F1Macro","MCC","Coverage"]]
            for col in ['BayesSearch','RandomSearch',"GridSearch"]:
                evalt = eval2[eval2['Type'] == col]
                plot_eval_metrics(evalt,lang=lang, text_col=text_col, col = col, datatype = r'\images\label_')

            eval.sort_values(by = ['Trial','k-fold','accuracy','MCC'], ascending = [True,True,False,False], inplace = True)
            eval.drop_duplicates(subset=['k-fold'], inplace=True, keep='first')
            plot_eval_folds(eval, lang=lang, text_col=text_col, datatype = r'\images\label_')

def laod_eval_data_classification():
    for lang in ['de']:#,'en']:
        for text_col in ['TOPIC']:#, 'URL_TEXT']:
            metrics = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\classification\pytorch_tuning_'+lang+r'\results\eval_results_'+text_col+r'.feather')
            
            metrics1 = metrics.sort_values(by = ['accuracy','MCC'], ascending = [False,False])
            metrics1.drop_duplicates(subset=['Type'], inplace=True, keep='first')
            metrics1 = metrics1[['Type',"Accuracy", "PrecisionMicro", "PrecisionMacro","F1Micro","F1Macro","MCC","Recall"]]
            for col in ['RandomSearch','BOHB', 'Hyperband']:
                eval = metrics1[metrics1['Type'] == col]
                plot_eval_metrics(eval,lang=lang, text_col=text_col, col = col, datatype = r'\images\classification_')

            metrics2 = metrics.sort_values(by = ['Trial','K-Fold','Accuracy','MCC'], ascending = [True,True,False,False], inplace = True)
            metrics2.drop_duplicates(subset=['K-Fold'], inplace=True, keep='first')
            plot_eval_folds(metrics2, lang=lang, text_col=text_col, datatype = r'\images\classification_')


def plot_coverage(df, lang, text_col):
    ind = np.arange(len(df['LF'].tolist()))
    width = 0.29

    fig, ax = plt.subplots(figsize=(8, 8))

    hbar1 = ax.barh(ind, df.Coverage, width, color='#478c2e', label='Coverage')
    ax.bar_label(hbar1, fmt='%.2f',fontsize=10)

    hbar2 = ax.barh(ind + width, df.Overlaps, width, color='#2e308c', label='Overlaps')
    ax.bar_label(hbar2, fmt='%.2f',fontsize=10)

    hbar3 = ax.barh(ind + 2*width, df.Conflicts, width, color='#8c2e2e', label='Conflicts')
    ax.bar_label(hbar3, fmt='%.2f',fontsize=10)
    
    ax.set(yticks=ind + width , yticklabels=df.LF, ylim=[2*width-1, len(df)], xlim = [0,1])

    # # Create offset transform by -145 in x direction
    # ax.set_yticklabels(df.LF, ha='left')
    # dx = -145/72.; dy = 0/72. 
    # offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # # apply offset transform to all y ticklabels. move +5 horizontal
    # for label in ax.yaxis.get_majorticklabels():
    #     label.set_transform(label.get_transform() + offset)

    ax.legend()
    # ax.set_title("TITLE")
    ax.set_xlabel(f"{df['k_fold'].tolist()[0]}-Fold {df['k_fold_split'].tolist()[0]}", fontweight='bold')

    # ax.margins(x = 0.5,y = 0)
    plt.subplots_adjust(left=0.28, bottom = 0.06, top = 0.99, right = 0.95)
    
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+r'\images\label_coverage_'+lang+'_'+text_col+r'.pdf')#pdf
    plt.close()

def plot_eval_folds(df, lang, text_col,datatype):
    barWidth = 0.25
    label = df['k-fold'].tolist()
    if 'label' in datatype:
        acc = 'accuracy'
    if 'classifi' in datatype:
        acc = 'Accuracy'

    r1 = np.arange(len(df['accuracy'].tolist()))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, df[acc].tolist(), color='#478c2e', width=barWidth, edgecolor='white', label='Accuracy')
    plt.bar(r2, df['MCC'].tolist(), color='#2e308c', width=barWidth, edgecolor='white', label='MCC')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('K-Fold', fontweight='bold')
    plt.xticks([r + barWidth-0.125 for r in range(len(df[acc].tolist()))], label)
    plt.ylim(0,1)
    # Create legend & Show graphic
    plt.legend()
    for i in range(len(label)):
        plt.text(i, round(df[acc].tolist()[i],2)+0.005, round(df[acc].tolist()[i],2), ha = 'center')
        plt.text(i+0.38, round(df['MCC'].tolist()[i],2)+0.005, round(df['MCC'].tolist()[i],2), ha = 'center')
    
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+r'kfolds_'+lang+'_'+text_col+r'.pdf')#pdf
    plt.close()

def plot_eval_metrics(df,lang,text_col,col, datatype):
    if 'label' in datatype:
        bars = ['Accuracy','MCC', 'Prec Mi', 'Prec Ma', 'F1 Mi', 'F1 Ma']
        x_pos = np.arange(len(bars))
        height = df['accuracy'].tolist() + df['MCC'].tolist() +df['PrecisionMicro'].tolist() +df['PrecisionMacro'].tolist()+df['F1Micro'].tolist()+df['F1Macro'].tolist()
        labels = ['Accuracy','Matthews Correlation', 'Precision Micro', 'Precision Macro', 'F1 Micro', 'F1 Macro']
        colors = ['#2e8c37','#2e308c','#8c2e2e','#8c7b2e','#2e8c75','#8c4f2e']
    if 'classifi' in datatype:
        bars = ['Accuracy','MCC', 'Precision Mi', 'Precision Ma', 'F1 Mi', 'F1 Ma', 'Re Ma']
        x_pos = np.arange(len(bars))
        height = df['Accuracy'].tolist() + df['MCC'].tolist() +df['PrecisionMicro'].tolist() +df['PrecisionMacro'].tolist()+df['F1Micro'].tolist()+df['F1Macro'].tolist()+df['Recall'].tolist()
        labels = ['Accuracy','Matthews Correlation', 'Precision Micro', 'Precision Macro', 'F1 Micro', 'F1 Macro', 'Recall Macro']
        colors = ['#2e8c37','#2e308c','#8c2e2e','#8c7b2e','#2e8c75','#8c4f2e','#478c2e']
    # Create bars
    plt.bar(x_pos, height, color=colors, label = labels)

    # Create names on the x-axis
    plt.xticks(x_pos, bars)
   
    # Create legend 
    plt.xlabel(col, fontweight='bold')
    plt.ylim(0,1)
    plt.legend()

    for i in range(len(bars)):
        plt.text(i, round(height[i],3)+0.01, round(height[i],3), ha = 'center')
    
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+r"metrics_"+col+"_"+lang+'_'+text_col+r'.pdf')#pdf
    plt.close()


# load_raw_data()
# load_clean_data()
# load_labeled_data()
load_eval_data_automated_label()

