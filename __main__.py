from src.topic_scrape.seed_scraper import *
from src.url_scrape.url_scrape.spiders import seeder
from src.cleans.clean import *
from src.cleans.nlp_lda import *
from src.automated_label.cluster_kmeans import *
from src.automated_label.label import *
from src.classifier import classifier_model_en as classifier_en
from src.classifier import classifier_model_ger as classifier_ger
#Main - Init a crawler with given searchlist Searchlist.xsls. Crawls and saves all information (run).

def select_language():
    print("Please select Language.\n\
        de: German \n\
        en: English")
    wrongInput = True
    while wrongInput:
        lang = str(input()).lower()
        if (str(input()).lower() == 'de' or str(input()).lower() == 'en'):
            wrongInput = False
    return lang

def main_menu():
    print("Welcome to Main Menu!\n\
        Please select an Option:\n\
            (1) Execute full process.\n\
            (2) Execute data crawling.\n\
            (3) Execute data cleaning.\n\
            (4) Execute topic extraction.\n\
            (5) Execute automated labeling.\n\
            (6) Execute classification. ")
    

if __name__ == "__main__":
    lang = select_language()  
    
    ###Get Top 10 Search Results per Keyword -> Save url in Seeder###
    start = time.process_time()
    scrape = TopicScraper("en",r'files\Seed.xlsx')
    scrape.run()

    ###Crawl data of given Seeder###
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

    ###Clean crawled data###
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

    ###Extract topics###
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

    ###Automated Labeling###

    ##Cluster k-Means with Classes##
    if lang == 'de':
        kmeans = TOPIC_KMeans(lang,r"files\topiced_classes.feather",r"files\topiced_texts.feather")
        kmeans.run()
    if lang == 'en': 
        kmeans = TOPIC_KMeans(lang,r"files\topiced_classes_en.feather",r"files\topiced_texts_en.feather")
        kmeans.run()

    ##Automated Labeling Model##
    if lang == 'de':
        pass
    if lang == 'en': 
        pass
    ###Classification###
    if lang == 'de':
        pass
    if lang == 'en': 
        pass


    print(time.process_time() - start)
