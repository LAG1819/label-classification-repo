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
        self.data = pd.read_json(df_path)
        
        self.text_col = 'URL_TEXT'
        self.url_col = 'URL'
        self.lang = lang

        self.target_path = t_path
    
    def regex_remove(self,row):
        output = []
        try:
            for text in row.split("|"):
                pattern_list = [r'^@.*\{.*\}', r'^\..*\{.*\}',r'\s\s+',r'\n',r'\xa0',r'dbx707', r'\xe2',r'\x80',r"\x8b", r"{{\.*}}"]# only digits: r'\b[0-9]+\b\s*'
                for pattern in pattern_list:
                    text = re.sub(pattern,'',text)
                #remove any word shorter than 3 characters
                out = re.sub(r'^\w{0,3}$','',text)

                output.append(out)
            output = list(filter(lambda x: len(x) > 3,output))
            output = list(set(list(filter(None,output))))
            #print(output)
            output = "|".join(output)
        except:
            output = ""
        return output
    
    def remove_nonText(self):
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.regex_remove(row))

    def flag_lang(self):
        def detect_language(text):
            return_lan = None
            try:
                return_lan = str(detect(text)).lower()
            except: 
                return_lan = None
            return return_lan
        
        self.data["LANG"]=self.data[self.text_col].apply(lambda row: detect_language(row))
        self.data = self.data[self.data["LANG"] == self.lang]
        self.data = self.data.reset_index(drop = True)

    def save_data(self):
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        self.data.to_feather(path)

    def run(self):
        self.remove_nonText()
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
    f = textFilter('de',r"files\raw_texts.json",r"files\cleaned_texts.feather")
    f.run()
    # f2 = textFilter('de',r"files\raw_topics.json",r"files\cleaned_topics.feather")
    # f2.run()