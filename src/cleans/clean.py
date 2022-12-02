from langdetect import detect
import pandas as pd
import os
import re
import pickle
import json

class textFilter:
    def __init__(self):
        # df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts.pkl"
        # file = open(df_path, 'rb')
        # data = pickle.load(file)
        # file.close()
        # # dict= pd.read_pickle(df_path)
        # # self.data = pd.DataFrame.from_dict(dict, orient = 'index').T

        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts.json"
        self.data = pd.read_json(df_path)
        
        self.text_col = 'URL-TEXT'
        self.url_col = 'URL'
    
    def regex_remove(self,row):
        output = []
        try:
            for text in row.split("|"):
                pattern_list = [r'^@.*\{.*\}', r'^\..*\{.*\}',r'\s\s+',r'\n',r'\xa0',r'dbx707', r'\xe2',r'\x80',r"\x8b"]# only digits: r'\b[0-9]+\b\s*'
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
                return_lan = detect(text)
            except: 
                return_lan = None
            return return_lan
        
        self.data["LANG"]=self.data[self.text_col].apply(lambda row: detect_language(row))

    def save_data(self):
        self.data.to_feather(str(os.path.dirname(__file__)).split("src")[0] + r"files\cleaned_texts.feather")

    def run(self):
        self.remove_nonText()
        self.flag_lang()
        self.save_data()
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
    f = textFilter()
    print(f.data)
    #f.run()