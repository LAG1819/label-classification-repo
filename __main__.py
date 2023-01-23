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
from src.cleans.nlp_lda import *
from src.automated_label.cluster_kmeans import *
from src.automated_label.label import *
# from src.classifier import classifier_model_en as classifier_en
# from src.classifier import classifier_model_ger as classifier_ger
from sys import exit


def select_language() -> str:
    """Selection of language based processing. Selection between german and english text possible.

    Returns:
        str: unicode of user selected language specification for text processing, labeling and classification. 
    """
    print("Please select Language.\n\
        de: German \n\
        en: English")
    wrongInput = True
    while wrongInput:
        lang = str(input()).lower()
        if lang == 'de' or lang == 'en': 
            wrongInput = False
    return lang

def end_session():
    """Menu exit and terminal close.
    """
    print("Finish Session! Unsaved changes will be deleted.")
    exit("Session ended.")

def execute_full_process(lang:str):
    """_summary_

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
    crawl_data(lang)
    clean_data(lang)
    extract_topics(lang)
    set_kMeans(lang)
    label_data(lang)
    classify_data(lang)

def crawl_data(lang:str):
    """_summary_

    Args:
        lang (str): unicode of language specification for text processing, labeling and classification
    """
     ###Get Top 10 Search Results per Keyword -> Save url in Seeder###
    scrape = TopicScraper("en",r'files\Seed.xlsx')
    scrape.run()

    ###Crawl data of given Seed###
    filenames =  str(os.path.dirname(__file__)).split("src")[0] + 'doc\scraping_'+lang+'.log'
    logging.basicConfig(filename=filenames, encoding='utf-8', level=logging.DEBUG)
    try:
        logging.info("[{log}]Crawling has started".format(log = datetime.now()))
        seeder.run_crawler(lang) 
    except Exception as e:
        logging.warning('[{log}]Crawling had been interrupted by error:{error}'.format(log = datetime.now(), error = e))
    finally:
        source_path = str(os.path.dirname(__file__)).split("src")[0]+r'files/raw_texts_new.json'
        class_path = str(os.path.dirname(__file__)).split("src")[0]+r'files/raw_classes.json'
        if os.path.exists(source_path):
            seeder.union_and_save_texts(lang)
        if os.path.exists(class_path):
            seeder.union_and_save_class(lang)

def clean_data(lang:str):
    if lang == 'de':
        d_class = textFilter('de',r"files\raw_classes.feather",r"files\cleaned_classes.feather")
        d_class.run()
        d = textFilter('de',r"files\raw_texts.feather",r"files\cleaned_texts.feather")
        d.run()
    if lang == 'en':
        e_class = textFilter('en',r"files\raw_classes_en.feather",r"files\cleaned_classes_en.feather")
        e_class.run()
        e = textFilter('en',r"files\raw_texts_en.feather",r"files\cleaned_texts_en.feather")
        e.run()

def extract_topics(lang:str):
    if lang == 'de':
        topics_d = TopicExtractor(2,r"files\cleaned_classes.feather",r"files\topiced_classes.feather",lang,True)
        topics_d.run()
        texts_d = TopicExtractor(2,r"files\cleaned_texts.feather",r"files\topiced_texts.feather", lang)
        texts_d.run()
    if lang == 'en': 
        texts_e = TopicExtractor(2,r"files\cleaned_texts_en.feather",r"files\topiced_texts_en.feather", lang)
        texts_e.run() 
        topics_e = TopicExtractor(2,r"files\cleaned_classes_en.feather",r"files\topiced_classes_en.feather",lang,True)
        topics_e.run()

def set_kMeans(lang:str):
     if lang == 'de':
        kmeans = TOPIC_KMeans(lang,r"files\topiced_classes.feather",r"files\topiced_texts.feather")
        kmeans.run()
     if lang == 'en': 
        kmeans = TOPIC_KMeans(lang,r"files\topiced_classes_en.feather",r"files\topiced_texts_en.feather")
        kmeans.run()


def label_data(lang:str):
    if lang == 'de':
        l = Labeler(r"files\topiced_texts.feather",r"files\labeled_texts.feather")
        l.run()
    if lang == 'en': 
        l = Labeler(r"files\topiced_texts_en.feather",r"files\labeled_texts_en.feather")
        l.run()


def classify_data(lang:str):
    if lang == 'de':
        pass
    if lang == 'en': 
        pass

def main_menu(lang:str):
    """Main Menu defined for external user.

    Args:
        lang (str): unicode of language to do apply whole process with
    """
    print("\n \
           ###################################################################################### \n \
           #######################-------------TALC---------------############################### \n \
           ##########-------------Topic based automated labeled classifer---------------######### \n \
           ###################################################################################### \n \
           Welcome to Main Menu!\n\
            Please select an Option:\n\
            (0) End session.\n\
            (1) Execute full process.\n\
            (2) Execute data crawling.\n\
            (3) Execute data cleaning.\n\
            (4) Execute topic extraction.\n\
            (5) Exercute automated labeling: k-Means. \n\
            (6) Execute automated labeling: Labeling.\n\
            (7) Execute classification. ")

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
        crawl_data()
    elif selected_execution == 3:
        clean_data()
    elif selected_execution == 4:
        extract_topics()
    elif selected_execution == 4:
        extract_topics()
    elif selected_execution == 6:
        label_data()
    elif selected_execution == 7:
        classify_data()


if __name__ == "__main__":
    lang = select_language()  
    main_menu(lang)
   
    # start = time.process_time()
    # print(time.process_time() - start)
