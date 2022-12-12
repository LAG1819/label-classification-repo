from langdetect import detect
import pandas as pd
import os
import re
import pickle
import json

class textFilter:
    def __init__(self,lang, path, t_path):
        # df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts.pkl"
        # file = open(df_path, 'rb')
        # data = pickle.load(file)
        # file.close()
        # # dict= pd.read_pickle(df_path)
        # # self.data = pd.DataFrame.from_dict(dict, orient = 'index').T

        df_path = str(os.path.dirname(__file__)).split("src")[0] + path
        self.data = pd.read_json(df_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        
        self.text_col = 'URL_TEXT'
        self.url_col = 'URL'
        self.lang = lang

        self.target_path = t_path
    
    def regex_remove(self,row):
        output = []
        sentence_lang = []
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

    def stopword_remove(self,row):
        output_sentence = []
        url = ["^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[äöüßa-zA-Z0-9()]{1,6}\\b(?:[-a-zäöüßA-Z0-9()@:%_\\+.~#?&\\/=]*)$"]
        email = ["^\S+@\S+\.\S+$"]
        zip = ["^[0-9]{5}(?:-[0-9]{4})?$"]
        phone = ["^\\+?[1-9][0-9]{7,14}$"]
        dates = ["^[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}$","^[0-9]{1,2}\\-[0-9]{1,2}\\-[0-9]{4}$"]
        website_stopwords = ['richtlinie\w*',"\w*recht\w* hinweis\w*","\w*recht\w*","\w*datenschutz\w*", "privacy","policy\w*","cooky\w*","cookie\w*","content\w*"," to ",\
                "anmelden",  "abmelden", "login","log in","logout", "log out", "kunden login", "online"," back ",\
                    "extras","news","report\w*","impressum","newsletter\w*", "owner","internet", "website\w*", "email\w*", "e-mail\w*", "mail\w*", "isbn", "issn",\
                        "produkte", "partner","übersicht", "veranstaltungen", "suche\w*","kauf\w*", "angebot\w*", "konfigur\w*",\
                            "zubehör", "garantie", "mehr", "modell\w*", "kontakt\w*", "skip",'useful links','link\w*',\
                                "buchen", "anfahrt", "finanzdienstleistung\w*" "connected", "required",\
                                    "englisch", "english","deutsch","german","google", "wikipedia", "navigation",\
                                        "januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember"]
                                       
        domain_stopwords = ["(g/km)","use case\w*"]
        numbers_only = ["^\\d+$","^\s?[0-9]+(\s+[0-9]+)*\s?$"]
        special_characters = ['[^äöüßA-Za-z0-9 ]+']
        short_words = ['^\w{0,3}$']
        all_stopwords = url+email+zip+phone+dates+website_stopwords+domain_stopwords+numbers_only+special_characters+short_words+numbers_only+[" \\d+ "]
        
        for sentence in row.split("|"):
            for pattern in all_stopwords:
                sentence = re.sub(pattern,'',sentence.lower())
                # if re.search(pattern, str(sentence).lower()):
            
            output_sentence.append(sentence)
        output_sentence = list(set(list(filter(None,output_sentence))))
        # print(output_sentence)
        return "|".join(output_sentence)
    
    def remove_nonText(self):
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.regex_remove(row))

    def remove_domainStopwords(self):
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.stopword_remove(row))

    def flag_lang(self):
        def detect_language(text):
            return_lan = None
            try:
                return_lan = str(detect(text)).lower()
            except: 
                return_lan = None
            return return_lan
        
        self.data["LANG"]=self.data[self.text_col].apply(lambda row: detect_language(row))
        # self.data = self.data[self.data["LANG"] == self.lang]
        self.data = self.data.reset_index(drop = True)

    def save_data(self):
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        self.data.to_feather(path)

    def run(self):
        self.remove_nonText()
        self.remove_domainStopwords()
        self.flag_lang()
        self.save_data()
        print(self.data.shape)
        print(self.data)
        print("Done Cleaning")

class urlFilter:
    def __init__(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts.csv"

    def flag_url(self):
        self.data["URL-FLAG"] = self.data[self.text_col]
        ###TO DO Define Rules to filter URL###

    def save_data(self):
        self.data.to_csv(os.path.join(self.package_dir,r'files\Output_texts.csv'), index = False)

if __name__ == "__main__":
    # f = textFilter('de',r"files\raw_texts.json",r"files\cleaned_texts.feather")
    # f.run()
    f2 = textFilter('de',r"files\raw_classes.json",r"files\cleaned_classes.feather")
    f2.run()

    
    # result = [re.sub("^\s?[0-9]+(\s+[0-9]+)*", "", w) for w in [ " 9 2015"," 2014", "00 88 00", "000 888 000"]]
    # print(result)