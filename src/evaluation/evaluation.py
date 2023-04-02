# <Evaluation of results of Automated Labeling and Textclassification (Process steps 4 and 5).>
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
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
import time
from datetime import datetime
import tarfile

def load_raw_data():
    """Load raw data and print data shapes.
    """
    raw_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_en.feather")
    raw_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\01_crawl\raw_texts_de.feather")
    print(raw_en.shape)
    print(raw_de.shape)

def load_clean_data():
    """Load clean data and print data shapes.
    """
    clean_en = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_en.feather")
    clean_de = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\02_clean\topiced_texts_de.feather")
    print(clean_en.shape)
    print(clean_de.shape)

def load_labeled_data():
    """Load labeled data and print data shapes.
    """
    for experiment in ['\Experiment1', '\Experiment2']:
        for lang in ['de', 'en']:
            print(experiment,":")
            
            labeled_TOPIC = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify"+experiment+"\labeled_texts_"+lang+r"_TOPIC.feather")
            print("TOPIC ",lang,labeled_TOPIC.shape)
            labeled_TOPIC['CATEGORY'] = labeled_TOPIC['LABEL'].replace({0:'AUTONOMOUS', 1:'CONNECTIVITY',2:'DIGITALISATION',3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"})
            plot_eval_distribution(labeled_TOPIC,lang=lang, text_col="TOPIC", data_path = r'\images\label'+experiment)
            
            labeled_URL_TEXT = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\04_classify"+experiment+"\labeled_texts_"+lang+r"_URL_TEXT.feather")
            print("URL_TEXT ",lang,labeled_URL_TEXT.shape)
            labeled_URL_TEXT['CATEGORY'] = labeled_URL_TEXT['LABEL'].replace({0:'AUTONOMOUS', 1:'CONNECTIVITY',2:'DIGITALISATION',3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"})
            plot_eval_distribution(labeled_URL_TEXT,lang=lang, text_col="URL_TEXT", data_path = r'\images\label'+experiment)  

def load_eval_data_automated_label():
    """Load results of labeled data and results of coverage of labeled data and generate graphics of each. Saves graphis in images folder.
    """
    for experiment in ['\Experiment1', '\Experiment2']:
        folder =str(os.path.dirname(__file__)).split("src")[0]+r'\images\label'+experiment
        if not os.path.exists(folder):
            os.makedirs(folder)
        for lang in ['de','en']:
            for text_col in ['TOPIC', 'URL_TEXT']:
                coverage = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label'+experiment+r'\model_tuning_'+lang+r'\results\coverage_results_'+text_col+r'.feather')
                coverage.sort_values(by = ['TRIAL','k_fold','k_fold_split','Overlaps','Conflicts','Coverage'], ascending = [False, False, False,True,True,False], inplace = True)
                coverage.drop_duplicates(subset=['LF'], inplace=True, keep='first')
                coverage['Polarity'] = coverage['Polarity'].replace({0:'AUTONOMOUS', 1:'CONNECTIVITY',2:'DIGITALISATION',3:"ELECTRIFICATION",4:"INDIVIDUALISATION",5:"SHARED",6:"SUSTAINABILITY"})
                coverage["LF"] = coverage['Polarity'] +"_"+coverage["LF"].str.split("_", n = 1, expand = True)[0] 
                plot_coverage(coverage, lang=lang, text_col=text_col, datatype = r'\images\label'+experiment)


                eval = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\label'+experiment+r'\model_tuning_'+lang+r'\results\eval_results_'+text_col+r'.feather')
                
                eval2 = eval.sort_values(by = ['accuracy'], ascending = [False])
                eval2.drop_duplicates(subset=['Type'], inplace=True, keep='first')
                eval2 = eval2[['Type',"accuracy", "PrecisionMicro", "PrecisionMacro","F1Micro","F1Macro","MCC","Coverage"]]
                eval2['Type'].replace({'BayesSearch':'Bayesian Optimization', 'RandomSearch':'Random Search',"GridSearch":"Grid Search"}, inplace=True)
                for col in ['Bayesian Optimization','Random Search',"Grid Search"]:
                    evalt = eval2[eval2['Type'] == col]
                    plot_eval_metrics(evalt,lang=lang, text_col=text_col, col = col, datatype = r'\images\label'+experiment)

                eval3 = eval.sort_values(by = ['accuracy','MCC'], ascending = [False,False])
                eval3.drop_duplicates(subset=['Type'], inplace=True, keep='first')
                eval3['Type'].replace({'BayesSearch':'Bayesian Optimization', 'RandomSearch':'Random Search',"GridSearch":"Grid Search"}, inplace=True)
                plot_grouped_eval_metrics(eval3[['Type',"accuracy", "PrecisionMicro", "PrecisionMacro","F1Micro","F1Macro","MCC"]], lang = lang, text_col=text_col, datatype=r'\images\label'+experiment)
                
                eval.sort_values(by = ['Trial','k-fold','accuracy'], ascending = [True,True,False], inplace = True)
                eval.drop_duplicates(subset=['k-fold'], inplace=True, keep='first')
                plot_eval_folds(eval, lang=lang, text_col=text_col, datatype = r'\images\label'+experiment)

def laod_eval_data_classification():
    """Load results of classfication data and generate graphics of each. Saves graphis in images folder.
    """
    for experiment in ['\Experiment1', '\Experiment2']:
        for lang in ['de','en']:
            for text_col in ['TOPIC']:
                metrics = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0]+r'models\classification'+experiment+r'\pytorch_tuning_'+lang+r'\results\eval_results_'+text_col+r'.feather')
                
                metrics1 = metrics.sort_values(by = ['Accuracy','MCC'], ascending = [False,False])
                metrics1.drop_duplicates(subset=['Type'], inplace=True, keep='first')
                metrics1 = metrics1[['Type',"Accuracy", "PrecisionMicro", "PrecisionMacro","F1Micro","F1Macro","MCC","RecallMacro"]]
                for col in ['Hyperband','BOHB']:
                    eval = metrics1[metrics1['Type'] == col]
                    plot_eval_metrics(eval,lang=lang, text_col=text_col, col = col, datatype = r'\images\classification'+experiment)

def plot_eval_distribution(df:pd.DataFrame,lang:str,text_col:str,data_path:str):
    """Plots the distribution of categories of labeled data of best results. Saves plot as pdf in images folder.

    Args:
        df (pd.DataFrame): Coverage dataframe containing results.
        lang (str): Language of Coverage dataframe results are based on. Can be englisch (en) or german (de).
        text_col (str): Columns of Coverage dataframe results are based on. Can be text (URL_TEXT) or topics (TOPIC)
        data_path (str): Folder-path to save evaluation result. Can be Automated Labeling (label) or Classification (classification).
    """
    try:
        a = df.loc[df['LABEL']==0]
        a = a['LABEL'].value_counts().values[0]
    except:
        a = 0
    try:
        c = df.loc[df['LABEL']==1]
        c = c['LABEL'].value_counts().values[0]
    except:
        c = 0
    try:
        d = df.loc[df['LABEL']==2]
        d = d['LABEL'].value_counts().values[0]
    except:
        d = 0
    try:
        e = df.loc[df['LABEL']==3]
        e = e['LABEL'].value_counts().values[0]
    except:
        e = 0
    try:
        i = df.loc[df['LABEL']==4]
        i = i['LABEL'].value_counts().values[0]
    except:
        i = 0
    try:
        sh = df.loc[df['LABEL']==5]
        sh = sh['LABEL'].value_counts().values[0]
    except:
        sh = 0
    try:
        su = df.loc[df['LABEL']==6]
        su = su['LABEL'].value_counts().values[0]
    except:
        su = 0

    bars = ['AUTONOMOUS', 'CONNECTIVITY','DIGITALISATION',"ELECTRIFICATION","INDIVIDUALISATION","SHARED","SUSTAINABILITY"]
    x_pos = np.arange(len(bars))
    height = [a,c,d,e,i,sh,su]
    labels = ['AUTONOMOUS', 'CONNECTIVITY','DIGITALISATION',"ELECTRIFICATION","INDIVIDUALISATION","SHARED","SUSTAINABILITY"]
    colors = ['#7a150d','#7a750d','#177a0d','#0d7a73','#0d0f7a','#7a0d75','#7a490d']
    
    # Create no legend with names on x-axis
    plt.bar(x_pos, height, color=colors)
    plt.xticks(x_pos, bars,rotation = 50)

    #Create legend without names on x-axis
    # plt.bar(x_pos, height, color=colors, label = labels)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
   
    # Create legend 
    plt.xlabel("Categories", fontweight='bold')
    plt.ylim(0,max(height)+800)
    plt.legend()

    for i in range(len(bars)):
        plt.text(i, round(height[i],3)+0.01, round(height[i],3), ha = 'center')
    
    plt.subplots_adjust(left=0.28, bottom = 0.25, top = 0.95, right = 0.95)
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+data_path+r"\class_distribution_"+lang+'_'+text_col+r'.pdf')
    plt.close()

def plot_coverage(df:pd.DataFrame, lang:str, text_col:str,datatype:str):
    """Plots Coverage of Labeling Functions of best result. Saves plot as pdf in images folder.

    Args:
        df (pd.DataFrame): Coverage dataframe containing results.
        lang (str): Language of Coverage dataframe results are based on. Can be englisch (en) or german (de).
        text_col (str): Columns of Coverage dataframe results are based on. Can be text (URL_TEXT) or topics (TOPIC)
    """
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

    ax.legend()
    
    ax.set_xlabel(f"Best Conflicts, Coverage and Overlaps in {df['k_fold'].tolist()[0]}-Fold Cross Validaiton", fontweight='bold', loc = "left")# and Fold {df['k_fold_split'].tolist()[0]}

    plt.subplots_adjust(left=0.28, bottom = 0.06, top = 0.99, right = 0.95)
    
    if not os.path.exists(str(os.path.dirname(__file__)).split("src")[0]+datatype):
        os.makedirs(str(os.path.dirname(__file__)).split("src")[0]+datatype)
        
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+'\label_coverage_'+lang+'_'+text_col+r'.pdf')
    plt.close()

def plot_eval_folds(df: pd.DataFrame, lang:str, text_col:str,datatype:str):
    """Plots best evaluation result of all iteration loops in optimization. Saves plot as pdf in images folder.

    Args:
        df (pd.DataFrame): Evaluation results dataframe.
        lang (str): Language of evaluation dataframe results are based on. Can be englisch (en) or german (de).
        text_col (str): Columns of evaluation dataframe results are based on. Can be text (URL_TEXT) or topics (TOPIC)
        datatype (str): Folder-path to save evaluation result. Can be Automated Labeling (label) or Classification (classification).
    """

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
    
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+r'\kfolds_'+lang+'_'+text_col+r'.pdf')
    plt.close()

def plot_eval_metrics(df:pd.DataFrame,lang:str,text_col:str,col:str, datatype:str):
    """Plots best evaluation results of all metrics of a selected parameteroptimization techinque. Saves plot as pdf in images folder.

    Args:
        df (pd.DataFrame): Evaluation results dataframe.
        lang (str): Language of evaluation dataframe results are based on. Can be englisch (en) or german (de).
        text_col (str): Columns of evaluation dataframe results are based on. Can be text (URL_TEXT) or topics (TOPIC)
        col (str): Selected hyperparameteroptimization technique, on which the best results shall be plotted with. Can be Random Search, Grid Search, Bayesian Search, Hyperband or BOHB.
        datatype (str): Folder-path to save evaluation result. Can be Automated Labeling (label) or Classification (classification).
    """
    if 'label' in datatype:
        bars = ['ACC','MCC', 'PRMI', 'PRMA', 'F1MI', 'F1MA']
        x_pos = np.arange(len(bars))
        height = df['accuracy'].tolist() + df['MCC'].tolist() +df['PrecisionMicro'].tolist() +df['PrecisionMacro'].tolist()+df['F1Micro'].tolist()+df['F1Macro'].tolist()
        labels = ['Accuracy','Matthews Correlation', 'Precision Micro', 'Precision Macro', 'F1 Micro', 'F1 Macro']
        colors = ['#2e8c37','#2e308c','#8c2e2e','#8c7b2e','#2e8c75','#8c4f2e']
    if 'classifi' in datatype:
        bars = ['ACC','MCC', 'PRMI', 'PRMA', 'F1MI', 'F1MA']
        x_pos = np.arange(len(bars))
        height = df['Accuracy'].tolist() + df['MCC'].tolist() +df['PrecisionMicro'].tolist() +df['PrecisionMacro'].tolist()+df['F1Micro'].tolist()+df['F1Macro'].tolist()#+df['RecallMacro'].tolist()
        labels = ['Accuracy','Matthews Correlation', 'Precision Micro', 'Precision Macro', 'F1 Micro', 'F1 Macro']
        colors = ['#2e8c37','#2e308c','#8c2e2e','#8c7b2e','#2e8c75','#8c4f2e']#,'#478c2e']
    # Create bars
    plt.bar(x_pos, height, color=colors)#, label = labels)

    # Create names on the x-axis
    plt.xticks(x_pos, bars)
   
    # Create legend 
    plt.xlabel(col, fontweight='bold')
    plt.ylim(0,1.1)
    plt.legend()

    for i in range(len(bars)):
        plt.text(i, round(height[i],3)+0.01, round(height[i],3), ha = 'center')
    
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+r"\metrics_"+col+"_"+lang+'_'+text_col+r'.pdf')#
    plt.close()

def plot_grouped_eval_metrics(df:pd.DataFrame,lang:str,text_col:str,datatype:str):
    """Plots best evaluation results of all metrics of a all parameteroptimization techinque. Saves plot as pdf in images folder.

    Args:
        df (pd.DataFrame): Evaluation results dataframe.
        lang (str): Language of evaluation dataframe results are based on. Can be englisch (en) or german (de).
        text_col (str): Columns of evaluation dataframe results are based on. Can be text (URL_TEXT) or topics (TOPIC)
        datatype (str): Folder-path to save evaluation result. Can be Automated Labeling (label).
    """

    barWidth = 0.7
    df.sort_values(by = ['Type'], ascending = [True], inplace = True)
    colors = ['#2e8c37','#2e308c','#8c2e2e','#8c7b2e','#2e8c75','#8c4f2e']
    bars = ['ACC','MCC', 'PRMI', 'PRMA', 'F1MI', 'F1MA']
    plt.rcParams["figure.figsize"] = (15,4)

    acc = df['accuracy'].tolist() 
    mcc = df['MCC'].tolist() 
    prmi = df['PrecisionMicro'].tolist() 
    prma= df['PrecisionMacro'].tolist()
    fmi = df['F1Micro'].tolist()
    fma = df['F1Macro'].tolist()
    bars8 = acc+mcc+prmi+prma+fmi+fma  

    # Set position of bar on X axis
    r1 = [0,5,10]
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r8 = r1+r2+r3+r4+r5+r6

    # Make the plot
    plt.bar(r1, acc, color=colors[0], width=barWidth, edgecolor='white', label=bars[0])
    plt.bar(r2, mcc, color=colors[1], width=barWidth, edgecolor='white', label=bars[1])
    plt.bar(r3, prmi, color=colors[2], width=barWidth, edgecolor='white', label=bars[2])
    plt.bar(r4, prma, color=colors[3], width=barWidth, edgecolor='white', label=bars[3])
    plt.bar(r5, fmi, color=colors[4], width=barWidth, edgecolor='white', label=bars[4])
    plt.bar(r6, fma, color=colors[5], width=barWidth, edgecolor='white', label=bars[5])
    
    # Text on the top of each bar
    label = [str(round(l,2)) for l in bars8]
    for i in range(len(r8)):
        plt.text(x = r8[i]-0.25 , y = bars8[i]+0.02, s = label[i], size = 10)    

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth+0.5 for r in r1], df['Type'].tolist()) 
    plt.ylim(0,1.0)
    plt.xlim(-0.5,14)
    plt.legend(loc = 'upper center', ncol = 3)
    plt.rcParams['axes.axisbelow'] = True

    plt.xlabel("Hyperparameter-Optimierungstechnicken", fontweight='bold')

    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+datatype+r"\metrics_HPOs_"+lang+'_'+text_col+r'.pdf')
    plt.close()

def plot_runtimes_automated_label():
    """Plots the manually given runtimes of the experiments of the automated labeling process.
    """
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (4,15)

    takl_topics_de = [63384,58622,58361]+[29476,24632,24932]
    takl_texts_de = [73531,64049,65506]+[25531,25642,25797]
    # takl_topics_en = [29476,24632,24932]
    # takl_texts_en = [25531,25642,25797]
    pakl_topics_de = [8930,8997,8752]+[7568,7462,7533]
    pakl_texts_de = [13819,13670,13814]+[8034,8190,8933]
    # pakl_topics_en = [7568,7462,7533]
    # pakl_texts_en = [8034,8190,8933]
    # takl_data = [takl_topics_de,takl_texts_de,takl_topics_en,takl_texts_en]
    # pakl_data = [pakl_topics_de,pakl_texts_de,pakl_topics_en,pakl_texts_en]
    data = [takl_topics_de,takl_texts_de,pakl_topics_de,pakl_texts_de]
    
    ax.boxplot(data)#, vert=False)#, showfliers=False)
    # x-axis labels
    ax.set_xticklabels(['TAKL_TOPICS', 'TAKL_TEXTS','PAKL_TOPICS', 'PAKL_TEXTS'])
    plt.xlabel('Laufzeiten', fontweight='bold')

    plt.rcParams['axes.axisbelow'] = True
    plt.show()
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+r"files\05_evaluation\images\label\Experiment2\pakl_runtime.pdf")

def plot_runtimes_classification():
    """Plots the manually given runtimes of the experiments of the classification process.
    """
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (4,15)

    takl_hyperband = [9060,16208]
    takl_bohb = [4673,9765]
    pakl_hyperband = [2610,12290]
    pakl_bohb = [7567,9472]
    
    data = [takl_hyperband,takl_bohb,pakl_hyperband,pakl_bohb]
    
    ax.boxplot(data)#, vert=False)#, showfliers=False)
    # x-axis labels
    plt.xlabel('Laufzeiten', fontweight='bold')
    ax.set_xticklabels(['TAKL_HYPERBAND', 'TAKL_BOHB','PAKL_HYPERBAND', 'PAKL_BOHB'])
    plt.rcParams['axes.axisbelow'] = True
    plt.show()
    plt.savefig(str(os.path.dirname(__file__)).split("src")[0]+r"files\05_evaluation\images\classification\metrics_runtime.pdf")

def plot_new_data_results_automated_label():
    """Plots the manually given validation and evaluation accuracies of the experiments of the automated labeling process.
    """
    # set width of bars
    barWidth = 0.5
    
    # set heights of bars
    accuracy_val = [0.44,0.36,0.41,0.34,0.49,0.27,0.55,0.29]
    accuracy_eval = [0,0,0,0,0.59,0.68,0.48,0.22]
    r3_val = accuracy_val+accuracy_eval
    
    # Set position of bar on X axis
    r1 = np.arange(len(accuracy_val)).tolist()
    r2 = [x + barWidth for x in r1]
    r3 = r1+r2
    
    # Make the plot
    plt.bar(r1, accuracy_val, color='#8c4f2e', width=barWidth, edgecolor='white', label='Validation')
    plt.bar(r2, accuracy_eval, color='#2e8c37', width=barWidth, edgecolor='white', label='Evaluation')

    # Text on the top of each bar
    label = [str(l) for l in r3_val]
    for i in range(len(r3)):
        plt.text(x = r3[i]-barWidth/2, y = r3_val[i]+0.02, s = label[i], size = 10)  
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth-0.2 for r in range(len(accuracy_val))], ['TAKL_TOPICS_DE','TAKL_TEXTS_DE','TAKL_TOPICS_EN','TAKL_TEXTS_EN','PAKL_TOPICS_DE','PAKL_TEXTS_DE','PAKL_TOPICS_EN','TAKL_TEXTS_EN'], rotation = 45)
    plt.ylim(0,1)
    plt.xlim(-0.3,7.8)
    # Create legend & Show graphic
    plt.xlabel('Evaluationsergebnisse', fontweight='bold')
    plt.legend()
    plt.show()

def plot_new_data_results_classification():
    """Plots the manually given validation and evaluation accuracies of the experiments of the classification process.
    """
    # set width of bars
    barWidth = 0.5
    
    # set heights of bars
    accuracy_val = [0.56,0.6,0.86,0.3]
    accuracy_eval = [0,0,0.61,0.09]
    r3_val = accuracy_val+accuracy_eval
    
    # Set position of bar on X axis
    r1 = np.arange(len(accuracy_val)).tolist()
    r2 = [x + barWidth for x in r1]
    r3 = r1+r2
    
    # Make the plot
    plt.bar(r1, accuracy_val, color='#8c4f2e', width=barWidth, edgecolor='white', label='Validation')
    plt.bar(r2, accuracy_eval, color='#2e8c37', width=barWidth, edgecolor='white', label='Evaluation')

    # Text on the top of each bar
    label = [str(l) for l in r3_val]
    for i in range(len(r3)):
        plt.text(x = r3[i]-0.1, y = r3_val[i]+0.02, s = label[i], size = 10)  
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth-0.25 for r in range(len(accuracy_val))], ['TAKL_TOPICS_DE','TAKL_TOPICS_EN','PAKL_TOPICS_DE','PAKL_TOPICS_EN'], rotation = 45)
    plt.ylim(0,1)
    plt.xlim(-0.3,3.8)
    # Create legend & Show graphic
    plt.xlabel('Evaluationsergebnisse', fontweight='bold')
    plt.legend()
    plt.show()

def calculate_runtime(start = (2023,3,1,22,18,5,2,9,0), end = (2023,3,2,15,54,29,5,2,8)):
    """Helper Function to analyze logging files. Can calculate time delta between two given timestemps.

    Args:
        start (tuple, optional): timestemp of start to subtract from end timestempt. Defaults to (2023,3,1,22,18,5,2,9,0).
        end (tuple, optional): timestempt of end where start timestemp get substracted of. Defaults to (2023,3,2,15,54,29,5,2,8).
    """
    start = datetime.strptime(time.asctime(start),"%a %b %d %H:%M:%S %Y")
    end = datetime.strptime(time.asctime(end),"%a %b %d %H:%M:%S %Y")
    dif = end - start
    print(dif.seconds)

def make_tarfile(output_filename:str, source_dir:str):
    """Compress a given folder as tar.gz file.

    Args:
        output_filename (str): File name of the compressed tar.gz.
        source_dir (str): Folder name of the folder to be compressed.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# load_raw_data()
# load_clean_data()
# load_labeled_data()
# load_eval_data_automated_label()
# laod_eval_data_classification()
# plot_runtimes_automated_label()
# plot_runtimes_classification()
# plot_new_data_results_automated_label()
# plot_new_data_results_classification()
# calculate_runtime((2023,3,16,8,45,0,4,2,1),(2023,3,16,11,13,50,7,6,1))
# make_tarfile("compressed_ml-classification-repo","ml-classification-repo")