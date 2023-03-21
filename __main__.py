# <Main Menu for the user. The whole process can be executed from here including several customizations by the user.>
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

# datapackage: All data is licensed under the Creative Common Attribution License as is the original data from geonames. https://creativecommons.org/licenses/by/4.0/legalcode
# All source code is licensed under the MIT licence. Further credits: https://github.com/lexman and https://okfn.org/ 

from src.topic_scrape.seed_scraper import *
from src.url_scrape.url_scrape.spiders import seeder
from src.cleans.clean import *
from src.cleans.topic_lda import *
from src.automated_label.cluster_kmeans import *
from src.automated_label.label import *
from src.classifier.classifier_model import run as classifier_run
from sys import exit


def select_language() -> str:
    """Selection of language based processing. Selection between german and english text possible.

    Returns:
        str:  unicode of language to run process with. It can be choosen between de (german) and en (englisch)
    """
    print("Please select Language.\n\
        1: German \n\
        2: English")
    wrongInput = True
    while wrongInput:
        selection = int(input())
        if selection == 1:
            lang = 'de'
            wrongInput = False
            return lang
        if selection == 2:
            lang = 'en'
            wrongInput = False
        else:
            print("Wrong Input. Please Selecet either language german (1) or english (2).")
    return lang
    

def select_database(nbr_execution:int) -> str:
    wrongPath = True
    df = None
    while wrongPath: 
        path = input("Please insert path (C:/User/absolute/path/data_folder/data.feather) to custom data. Supported extensions right now are [.feather, .xlsx, .csv]:")
        if os.path.exists(path):
            wrongPath = False
            try:        
                if path.lower().endswith(('.feather')):
                    df = pd.read_feather(os.path.join(os.path.dirname(__file__), path))
                if path.lower().endswith(('.xlsx')):
                    df = pd.read_excel(os.path.join(os.path.dirname(__file__), path), header = 0)
                if path.lower().endswith(('.csv')):
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), path), header = 0)
                path_to_save = str(os.path.dirname(__file__)) + r"\files\custom_data.feather"
                df.to_feather(path_to_save)
                return path_to_save
            except Exception as e:
                print("Something went wrong while reading custom file", e)
                
                print("Do you still want to perform process on custom data?(y/n)")
                wrongInput = True
                while wrongInput:
                    wrongInput = False
                    selection = input()
                    if selection != "y" or selection != "n":
                        wrongInput=True
                        print("Do you still want to perform process on custom data?(y/n)")
                
                if selection == "n":
                    return
                wrongPath = True            
        else:
            wrongPath = True
            print("Path could not be found. Please retry!")
            
           

def end_session():
    """Menu exit and terminal close.
    """
    print("Finish Session! Unsaved changes will be deleted.")
    exit("Session ended.")

def execute_full_process(lang:str):
    """Performs complete process, starting from data mining, through data cleaning, labeling and development of label model and text classification model.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    crawl_data(lang)
    clean_data(lang)
    extract_topics(lang)
    print("Plase check data and identify classes!")
    print("Classes identified? (y/n)")
    wrongAnswer = True
    while wrongAnswer:
        decision = str(input())
        if decision == "y":
            wrongAnswer = False   
    label_data(lang)
    classify_data(lang)

def crawl_data(lang:str, data_path:str=None):
    """Runs a combination of two WebCrawlers to crawl web pages. The seed of the WebCrawlers is defined in Seed.xlsx.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used. 
    """
     ###Get Top n Search Results per Keyword -> Save url in Seed.feather###
    print("Topic Scraping started.")
    if data_path:
        TopicScraper(lang,s_path = data_path).run()
    else: 
        TopicScraper(lang).run()

    print("Seed Scraping started.")
    ###Crawl data of given Seed###
    seeder.crawl_data(lang)

def clean_data(lang:str, data_path:str = None):
    """Performs a cleanup of the text data of the previously crawled web pages or other specified database.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used.
    """
    if os.path.exists(data_path):
        path = data_path.split("ml-classification-repo\\")[-1]
        textFilter(lang = lang,s_path = path).run()
    else:
        textFilter(lang = lang).run()
    
def extract_topics(lang:str, data_path:str = None):
    """Performs the extraction of topics based on the Latent Dirichlet Allocation algorithm

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used.
    """
    if os.path.exists(data_path):
        path = data_path.split("ml-classification-repo\\")[-1]
        TopicExtractor(s_path = path,lang = lang).run()
    else:
        TopicExtractor(lang = lang).run()
      
def generate_kMeans(lang:str, data_path:str = None):
    """Trains a k-Means cluster with k=number of classes (user defined). The centroids are fixed points. The data basis of the centroids can be customized in Seed.xlsx.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used.
    """
    if os.path.exists(data_path):
        path = data_path.split("ml-classification-repo\\")[-1]
        textFilter(lang = lang,s_path = path,t_path = r"files\cleaned_classes_"+lang+r".feather").run()
    else:
        seeder.crawl_centroids(lang)
        textFilter(lang = lang,s_path = r"files\raw_classes_"+lang+r".feather",t_path = r"files\cleaned_classes_"+lang+r".feather").run() 

    TopicExtractor(input_topic = 2,s_path = r"files\cleaned_classes_"+lang+r".feather",t_path = r"files\topiced_classes_"+lang+r".feather",lang = lang,zentroid = True).run()
    TOPIC_KMeans(lang = lang , data_path= r"files\topiced_texts_"+lang+r".feather",topics_path = r"files\topiced_classes_"+lang+r".feather").run()


def label_data(lang:str, data_path:str = None):
    """Performs labeling of input data and development of a label model based on the input data.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used.
    """
    wrongNumber = True
    while wrongNumber:
        wrongNumber = False
        print("Please select type of data labeling:\n\
          1: Partial data labeling. \n\
          2: Total data labeling.")
        number = input()
        if number != 1 or number != 2:
            wrongNumber = True
    if number == 1:
        if os.path.exists(data_path):
            path = data_path.split("ml-classification-repo\\")[-1]
            Labeler(lang = lang,s_path = path, partial = False).run()
            Labeler(lang = lang,s_path = path, column = 'URL_TEXT', partial = False).run()
        else:
            Labeler(lang = lang, partial = False).run()
            Labeler(lang = lang, column = 'URL_TEXT', partial = False).run()
    elif number == 2:
        generate_kMeans(lang=lang)
        if os.path.exists(data_path):
            path = data_path.split("ml-classification-repo\\")[-1]
            Labeler(lang = lang,s_path = path).run()
            Labeler(lang = lang,s_path = path, column = 'URL_TEXT').run()
        else:
            Labeler(lang = lang).run()
            Labeler(lang = lang, column = 'URL_TEXT').run()


def classify_data(lang:str, data_path:str):
    """Performs development and training of a BERT-based text classifier based on labeled input data. 

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        data_path(str): Selected path to the data to be used.
    """
    if os.path.exists(data_path):
        path = path = data_path.split("ml-classification-repo\\")[-1]
        classifier_run(lang = lang, data_path = path,col = 'TOPIC')
    else: 
        classifier_run(lang = lang, col = 'TOPIC')

def main_menu(lang:str):
    """Main Menu defined for external user.

    Args:
        lang (str): unicode of language to do apply whole process with
    """
    print("\n \
           ###################################################################################### \n \
           #######################-------------TALC---------------############################### \n \
           ##########--------Topic based automated labeler and text classifer----------########## \n \
           ######################################################################################")
    print("\n \
            Please select an Option:\n\
            (0) End session.\n\
            (1) Execute full process.\n\
            (2) Execute data crawling.\n\
            (3) Execute data cleaning.\n\
            (4) Execute topic extraction.\n\
            (5) Execute automated labeling: k-Means. \n\
            (6) Execute automated labeling: Train and Apply Label Modell.\n\
            (7) Execute classification: Train and Apply Classification Modell. ")

    wrongInput = True
    while wrongInput:
        selected_execution = int(input())
        if selected_execution < 6:
            wrongInput = False
        else:
            print("Wrong Input. Please retry!")

    if selected_execution == 0:
        end_session()
    elif selected_execution == 1:
        print("Full Process will be executed.")
        print("WARNING: Execution may require several hours! Still proceed? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            answer = str(input())
            if answer == 'y' or answer == 'n':
                wrongAnswer = False
            else:
                print("Wrong input! Please retry!")

        if answer == 'y':
            execute_full_process()
        else:
            main_menu(lang)

    elif selected_execution == 2:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(2)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        
        crawl_data(lang, data_path=data_path)
        
    elif selected_execution == 3:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(3)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        clean_data(lang,data_path=data_path)

    elif selected_execution == 4:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(4)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        extract_topics(lang, data_path=data_path)

    elif selected_execution == 5:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(5)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        generate_kMeans(lang, data_path=data_path)
        
    elif selected_execution == 6:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(6)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        label_data(lang,data_path=data_path)

    elif selected_execution == 7:
        data_path = None
        print("Take custom data? (y/n)")
        wrongAnswer = True
        while wrongAnswer:
            decision = str(input())
            if decision == "y":
                data_path = select_database(7)
                wrongAnswer = False
            elif decision == "n":
                wrongAnswer = False
            else: 
                print("Wrong input. Take custom data? (y/n)")
        classify_data(lang,data_path=data_path)


if __name__ == "__main__":
    lang = select_language()
    main_menu(lang)
    #C:/Users/Luisa/Documents/DataScience_M.Sc._HDM/Masterarbeit/Repository/ml-classification-repo/files/01_crawl/pristine_raw_texts_de.feather

    # t = os.path.exists(r"C:\Users\Luisa\Documents\DataScience_M.Sc._HDM\Masterarbeit\Repository\ml-classification-repo\files\01_crawl\pristine_raw_texts_de.feather")
    # # save raw data that are not already in the all data 
    # lang = 'en'
    # df_eval_data = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r'\files\01_crawl\raw_texts_'+lang+'.feather')
    # df_all_data = pd.read_feather(str(os.path.dirname(__file__)).split("src")[0] + r'\files\01_crawl\alldata_raw_texts_'+lang+'.feather')

    # df3 = df_eval_data.merge(df_all_data, on='URL', how='left', indicator=True)
    # df = df3.loc[df3['_merge'] == 'left_only', 'URL']
    # df_eval_data_only = df_eval_data[df_eval_data['URL'].isin(df)]
    # print("Shape eval with overlaps: ", df_eval_data.shape[0])
    # print("Shape absolutely new: ", df_eval_data_only.shape[0])
    # df_eval_data_only.reset_index(inplace =True)
    # df_eval_data_only.to_feather(str(os.path.dirname(__file__)).split("src")[0] + r'\files\01_crawl\pristine_raw_texts_'+lang+'.feather')
