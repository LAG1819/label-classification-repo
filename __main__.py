# <one line to give the program's name and a brief idea of what it does.>
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
from src.classifier import classifier_model
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
        selection = input()
        if selection == 1:
            lang = 'de'
            wrongInput = False
        elif selection == 2:
            lang == 'en'
            wrongInput = False
        else:
            print("Wrong Input. Please Selecet either language german (1) or english (2).")
    return lang

def select_database() -> str:
    print("Do you want to perform process on custom data?(y/n)")
    wrongInput = True
    while wrongInput:
        wrongInput = False
        selection = input()
        if selection != "y" or selection != "n":
            wrongInput=True
    if selection == "y":
        "Please insert absolute path (str) to custom data. Supported extensions right now are (.feather, .xlsx, .csv):"
        wrongPath = True
        df = None
        while wrongPath:
            wrongPath = False
            path = str(input())
            if not os.path.exists(path):
                wrongInput=True
                print("Path could not be found. Please retry!")
            else:
                try:        
                    if path.lower().endswith(('.feather')):
                        df = pd.read_feather(path)
                    if path.lower().endswith(('.xlsx')):
                        df = pd.read_excel(path, header = 0)
                    if path.lower().endswith(('.csv')):
                        df = pd.read_csv(path, header = 0)
                    path_to_save = str(os.path.dirname(__file__)) + r"files\custom_data.feather"
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
    print("Plase check cata and identify classes!")
    print("Classes identified? (y/n)")
    wrongAnswer = True
    while wrongAnswer:
        decision = str(input())
        if decision == "y":
            wrongAnswer = False
   
    label_data(lang)
    classify_data(lang)

def crawl_data(lang:str, number = 0):
    """Runs a combination of two WebCrawlers to crawl web pages. The seed of the WebCrawlers is defined in Seed.xlsx.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
        number(int): Selected Number of first Google search hits for a keyword. 
    """
     ###Get Top 10 Search Results per Keyword -> Save url in Seed.feather###
    TopicScraper(lang,r'files\Seed.xlsx', number).run()

    ###Crawl data of given Seed###
    seeder.crawl_data(lang)

def clean_data(lang:str, data_path:str):
    """Performs a cleanup of the text data of the previously crawled web pages or other specified database.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    if os.path.exists(data_path):
        textFilter(lang,data_path,r"files\cleaned_texts_"+lang+r".feather").run()
    else:
        textFilter(lang,r"files\raw_texts_"+lang+r".feather",r"files\cleaned_texts_"+lang+r".feather").run()
    
def extract_topics(lang:str, qty_topics =2):
    """Performs the extraction of topics based on the Latent Dirichlet Allocation algorithm

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    TopicExtractor(qty_topics,r"files\cleaned_texts_"+lang+r".feather",r"files\topiced_texts_"+lang+r".feather", lang).run()
      
def generate_kMeans(lang:str):
    """Trains a k-Means cluster with k=number of classes (user defined). The centroids are fixed points. The data basis of the centroids can be customized in Seed.xlsx.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    seeder.crawl_centroids(lang)
    textFilter(lang,r"files\raw_classes_"+lang+r".feather",r"files\cleaned_classes_"+lang+r".feather").run()
    TopicExtractor(2,r"files\cleaned_classes_"+lang+r".feather",r"files\topiced_classes_"+lang+r".feather",lang,True).run()
    TOPIC_KMeans(lang,r"files\topiced_classes_"+lang+r".feather",r"files\topiced_texts_"+lang+r".feather").run()


def label_data(lang:str):
    """Performs labeling of input data and development of a label model based on the input data.

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
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
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_TOPIC'+".feather",'TOPIC', False)
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_URL_TEXT'+".feather",'URL_TEXT', False)
    elif number == 2:
        generate_kMeans(lang)
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_TOPIC'+".feather",'TOPIC', True)
        Labeler(lang,r"files\02_clean\topiced_texts_"+lang+".feather",r"files\04_classify\labeled_texts_"+lang+'_URL_TEXT'+".feather",'URL_TEXT', True)


def classify_data(lang:str):
    """Performs development and training of a BERT-based text classifier based on labeled input data. 

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    classifier_model.run(lang, col = 'TOPIC')

def main_menu(lang:str, data_path:str):
    """Main Menu defined for external user.

    Args:
        lang (str): unicode of language to do apply whole process with
        data_path (str): path to custom database.
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
            wrongAnswer = False
            answer = str(input())
            if answer != 'y' or answer != 'n':
                wrongAnswer = True
                print("Wrong input! Please retry!")

        if answer == 'y':
            execute_full_process()
        else:
            main_menu(lang)

    elif selected_execution == 2:
        crawl_data(lang)
    elif selected_execution == 3:
        clean_data(lang, data_path)
    elif selected_execution == 4:
        extract_topics(lang, data_path)
    elif selected_execution == 5:
        generate_kMeans(lang)
    elif selected_execution == 6:
        label_data(lang, data_path)
    elif selected_execution == 7:
        classify_data(lang, data_path)


if __name__ == "__main__":
    lang = select_language()
    data_path = select_database()  
    main_menu(lang, data_path)
   
    # start = time.process_time()
    # print(time.process_time() - start)
