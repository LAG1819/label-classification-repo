from langdetect import detect
import pandas as pd
import os
import re
import pickle
import json
import spacy
nlp = spacy.load("de_dep_news_trf") # trained on bert based german cased

class textFilter:
    """Class to clean and filter raw texts (raw_texts.json) that had been crawled in previous step. 
    """
    def __init__(self,lang:str, path:str, t_path:str):
        """Initialise a textFilter object that can clean raw html text.

        Args:
            lang (str): unicode of language to filter raw texts only in that language 
            path (str): source path to file containing raw texts to clean
            t_path (str): target path to save file with cleaned texts
        """
        # df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts.pkl"
        # file = open(df_path, 'rb')
        # data = pickle.load(file)
        # file.close()
        # # dict= pd.read_pickle(df_path)
        # # self.data = pd.DataFrame.from_dict(dict, orient = 'index').T

        self.text_col = 'URL_TEXT'
        self.url_col = 'URL'
        self.lang = lang

        df_path = str(os.path.dirname(__file__)).split("src")[0] + path
        self.data = pd.read_feather(df_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        #self.data = self.data.head(4)
        self.cities = self.load_cities(self.lang)

        self.target_path = t_path

    def load_cities(self,lang:str) -> dict:
        """Load a complete list of citynames of a country based on language unicode. It can be choosen between german(de) and english(en)

        Args:
            lang (str): unicode of language to select country with citynames with dedicated language

        Returns:
            dict: Return dictionary containing all city names as key and the string to replace it with as value.
        """
        cites_dic = {}
        if lang == 'de':
            city_path = str(os.path.dirname(__file__)).split("src")[0] + r"files/germany.json"
            german_cites = pd.read_json(city_path)["name"].tolist()
            for city in german_cites:
                cites_dic[city.lower()] = ""  
        return cites_dic
    
    def regex_remove(self,row:str) -> str:
        """Basic text cleaning with help of regex (rowwise). Removes all text that is part of scripting like html or xml.

        Args:
            row (str): raw text of one sample. Each sample contains text of one crawled website.  

        Returns:
            str: pre cleaned text of on sample.
        """
        output = []
        
        xml = ["(?:<from.*?>)(.*?)(?:<\\/from>)"]
        html = ["<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"]
        try:
            #04:57,'07-05-2018'
            for sentence in row.split("|"):
                pattern_list = xml+html+[r'^@.*\{.*\}', r'^\..*\{.*\}',r'\s\s+',r'\n',r'\xa0',r'dbx707', r'\xe2',r'\x80',r"\x8b", r"{{\.*}}", r"\x9d", r"\u200b"]# only digits: r'\b[0-9]+\b\s*'
                for pattern in pattern_list:
                    sentence = re.sub(pattern,'',sentence)
                #remove any word shorter than 3 characters
                out = re.sub(r'^\w{0,3}$','',sentence)
                output.append(out)
            
            #output = list(filter(lambda x: len(x) > 3,output))
            output = list(set(list(filter(None,output))))
            
            output = "|".join(output)
    
        except Exception as e:
            print(e)
            output = ""
        return output

    def stopword_remove(self,row:str) -> str:
        """Advanced text cleaning with help of regex (rowwise). Removes advanced websites specific "stopwrods" as well as domain specific "stopwrods" as 
        well as other basic stopwrods to remove. 

        Args:
            row (str): pre cleaned text of one sample. Each sample contains text of one crawled website.  

        Returns:
            str: full cleaned text of on sample.
        """
        output_sentence = []
        url = ["^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[äöüßa-zA-Z0-9()]{1,6}\\b(?:[-a-zäöüßA-Z0-9()@:%_\\+.~#?&\\/=]*)$", "www\w*de","www\w*com"]
        email = ["^\S+@\S+\.\S+$"]
        zip = ["^[0-9]{5}(?:-[0-9]{4})?\s?\w*$"]
        phone = ["^\\+?[1-9][0-9]{7,14}$"]
        dates = ["^[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}$","^[0-9]{1,2}\\-[0-9]{1,2}\\-[0-9]{4}$", "^[0-9]{4}\\-[0-9]{1,2}\\-[0-9]{1,2}$"]
        website_stopwords = ["explore","allgemeine geschäftsbedingungen","allgemein\*",'richtlinie\w*',"\w*recht\w* hinweis\w*","\w*recht\w*","\w*datenschutz\w*", "privacy","policy\w*","cooky\w*","cookie\w*","content\w*"," to ",\
                "anmeld\w*",  "abmeld\w*", "login","log in","logout", "log out", "kunden login", "online","zurück","back","start","select\w*", "ausw\w*","close",\
                    "extras","news","report\w*","impressum","newsletter\w*", "owner","internet", "website\w*", "email\w*", "e-mail\w*", "mail\w*", "isbn", "issn",\
                        "produkte", "partner","übersicht", "veranstaltungen", "suche\w*","kauf\w*", "angebot\w*", "konfigur\w*", "configur\w*","nutzer\w*","icon\w*",\
                            "zubehör", "garantie", "mehr", "modell\w*", "kontakt\w*","contact\w*","anfrage\w*","skip",'useful links','link\w*',"pin\w*","passw\w*", "password\w*",\
                                "buchen","book" "anfahrt", "finanzdienstleistung\w*" "connected", "required", "sitemap\w*", "\w*\s?abo\w*", 'social media', "socialmedia",\
                                    "englisch", "english","deutsch","german","google", "wikipedia", "navigation","\w*shop\w*", "\w*magazin\w*", "lifestyle",\
                                        "facebook\w*", "youtube\w*","instagram\w*","xing\w*","linkedin\w*", "blog\w*","spiegel\w*","twitter\w*","sms","video"\
                                            "archiv\w*", "artikel\w*", "article\w*","side\w*", "seite\w*","site","app\w*","\s?abgerufen\s?\w*\s*\d*",\
                                                "januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember",\
                                                    "dbx707", "db11","\w*\s?straße\s?\d*","\w*\s?strasse\w*", "tel\w*", "\w*\s?download\w*",\
                                                        "covid\w*\s?\d*", "corona\w*\s?\d*"]
                                       
        domain_stopwords = ["(g/km)","use case\w*", "unternehme\w*", "gmbh", "cokg", "co kg", "consult\w*", "handel\w*", "händler\w*", "leistung\w*"]
        numbers_only = ["^\\d+$","^\s?[0-9]+(\s+[0-9]+)*\s?$", "\(.*\)","\[.*\]", "^\d+.\d+"," \\d+ "]
        special_characters = ['[^äöüßA-Za-z0-9 ]+']#['[\(,.:\);^]']
        short_words = ['^\w{0,3}$', '^\s+']
        all_stopwords = url+email+zip+phone+dates+numbers_only+special_characters+website_stopwords+domain_stopwords+short_words
        
        for sentence in row.split("|"):
            out_sentence = []
            for word in sentence.split(" "):
                for pattern in all_stopwords:
                    word = re.sub(pattern,'',word.lower())
                    # if re.search(pattern, str(sentence).lower()):
                out_sentence.append(word.lstrip())
            out_sentence = list(set(list(filter(None,out_sentence))))
            if out_sentence:
                output_sentence.append(" ".join(out_sentence))
        #print(" ".join(output_sentence))
        return " ".join(output_sentence)
    
    def remove_nonText(self):
        """Apply rowwise basic text cleaning with regex_remove() on raw texts.
        """
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.regex_remove(row))

    def remove_domainStopwords(self):
        """Apply rowwise advanced text cleaning with stopword_remove() on pre cleaned texts.
        """
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.stopword_remove(row))

    def remove_cityNames(self):
        """Removes all city names in text
        """
        regex = re.compile("|".join(map(re.escape, self.cities.keys(  ))))
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: regex.sub(lambda match: self.cities[match.group(0)], row))

    def lemmatize_text(self):
        """Lemmatize text with help of spacy 
        """
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: " ".join([token.lemma_ for token in nlp(row)]))

    def flag_lang(self):
        """Detect Language of each sample (row) containing text of one crawled website. 
        """
        def detect_language(text:str) -> str:
            """Helper Function for detecting language of row using langdetect package.
            Args:
                text (str): one sample (row) containing text of one crawled website.

            Returns:
                str: detected language as unicode.
            """
            return_lan = None
            try:
                return_lan = str(detect(text)).lower()
            except: 
                return_lan = None
            return return_lan
        
        self.data["LANG"]=self.data[self.text_col].apply(lambda row: detect_language(row))
        if self.lang != None:
            self.data = self.data[self.data["LANG"] == self.lang]
        self.data = self.data.reset_index(drop = True)

    def save_data(self):
        """Save cleaned texts to target path. 
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

    def run(self):
        """Run function of textFilter class. Firstly a basic text cleansing will be applied, than an advanced text cleaning and advanced stopwords removal will be applied.
        Than the language of cleaned texts is applied and all texts that are not matching the initialised lang are filtered. In advance all city 
        names in texts will be removed and text will be lemmatized.
        Filtered and cleaned texts are finally saved to target path. 
        """
        self.remove_nonText()
        self.remove_domainStopwords()
        print("Non textual elements and stopwords had been removed.")
        self.flag_lang()
        print("Languages had been detected and filtered.")
        self.remove_cityNames()
        print("City names had been removed.")
        self.lemmatize_text()
        print("Text had been lemmatized.")
        self.save_data()
        print(self.data.shape)
        print(self.data)
        print("Done Cleaning.")

if __name__ == "__main__":
    f = textFilter('de',r"files\raw_texts.feather",r"files\cleaned_texts.feather")
    f.run()
    # f2 = textFilter(None,r"files\raw_classes.feather",r"files\cleaned_classes.feather")
    # f2.run()
    
    # result = [re.sub("\s?abgerufen\s?\w*\s*\d*", "", w) for w in [ "abgerufen am 2709","abgerufen2709", " abgerufen am27 09", "000 888 000"]]
    # print(result)