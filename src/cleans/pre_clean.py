from langdetect import detect
import pandas as pd
import os
import re

class textFilter:
    def __init__(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts.csv"
        self.data = pd.read_csv(df_path, header = 0, delimiter=",")
        self.text_col = 'TEXT'
        self.url_col = 'URL'
    
    def regex_remove(self,row):
        output = []
        for text in row.split("|"):
            out = re.sub(r'^@.*\{.*\}','',text)
            out = re.sub(r'^\..*\{.*\}','',out)
            out = re.sub(r'\s\s+',' ',out)
            out = re.sub(r'\n','',out)
            out = re.sub(r':\xa0','',out)

            output.append(out)
        output = list(filter(None,output))

        return "|".join(output)
    
    def remove_nonText(self):
        self.data[self.text_col] = self.data[self.text_col].apply(lambda row: self.regex_remove(row))

    def flag_lang(self):
        self.data["LANG"]=self.data[self.text_col].apply(lambda row: detect(row))

    def save_data(self):
        self.data.to_csv(str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_pre_clean.csv", index = False)

    def run(self):
        self.flag_lang()
        self.remove_nonText()
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
    f.run()
    print(f.data['LANG'])