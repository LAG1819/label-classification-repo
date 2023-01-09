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

from langdetect import detect
import pandas as pd
import os
import re
import datapackage
import spacy
import math
import logging 
from datetime import datetime

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
        self.target_path = t_path
        
        self.data = self.load_data(path, t_path)
        self.cities = self.load_cities()
        
        filenames =  str(os.path.dirname(__file__)).split("src")[0] + 'doc\cleaning_'+lang+'.log'
        logging.basicConfig(filename=filenames, encoding='utf-8', level=logging.DEBUG)

    def load_data(self, path:str, t_path:str) -> pd.DataFrame:
        """Loads raw dataset containing all data samples that not had been already cleaned.

        Args:
            path (str): source path to file containing raw texts to be cleaned
            t_path (str): target path to save file with cleaned texts

        Returns:
            pd.DataFrame: DataFrame with raw dataset excluding all samples that had been already cleaned
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + path
        data = pd.read_feather(df_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        data = data[data['URL_TEXT']!=""]

        df_t_path = str(os.path.dirname(__file__)).split("src")[0] + t_path
        if os.path.exists(df_t_path):
            cleaned_data = pd.read_feather(df_t_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        else:
            cleaned_data = pd.DataFrame(columns=data.columns.tolist())

        # Identify what values are in raw data and not already in cleaned data
        left_join = data.merge(cleaned_data, on='URL', how='left', indicator=True)
        left_join_df = left_join.loc[left_join['_merge'] == 'left_only', 'URL']
        raw_data = data[data['URL'].isin(left_join_df)]
        if "LANG" in raw_data.columns:
            raw_data = raw_data.drop(columns="LANG", axis = 1)
        print("Raw data (total):", data.shape)
        print("Raw data:", raw_data.shape)
        print("Cleaned data:", cleaned_data.shape)
        return raw_data

    def load_cities(self) -> dict:
        """Load a complete list of citynames of a cities above 15,000 inhabitants. All data is licensed under the Creative Common 
        Attribution License as is the original data from geonames.

        Returns:
            dict: Return dictionary containing all city names as key and the string to replace it with as value.
        """
        cites_dic = {}

        city_data = 'https://datahub.io/core/world-cities/datapackage.json'
         # to load Data Package into storage
        package = datapackage.Package(city_data)

        # to load only tabular data
        resources = package.resources
        for resource in resources:
            if resource.tabular:
                data = pd.read_csv(resource.descriptor['path'])['name'].tolist()
                for city in data:
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
        return " ".join(output_sentence)
    
    def remove_nonText(self, input_data:pd.DataFrame) -> pd.DataFrame:
        """Apply rowwise basic text cleaning with regex_remove() on raw texts.

        Args:
            input_data (pd.DataFrame): DataFrame containing chunk of samples.

        Returns:
            pd.DataFrame: DataFrame with edited chunk of samples.
        """
        data = input_data.copy()
        data[self.text_col] = data[self.text_col].apply(lambda row: self.regex_remove(row))
        return data

    def remove_domainStopwords(self, input_data:pd.DataFrame) -> pd.DataFrame:
        """Apply rowwise advanced text cleaning with stopword_remove() on pre cleaned texts.

        Args:
            input_data (pd.DataFrame): DataFrame containing chunk of samples.

        Returns:
            pd.DataFrame: DataFrame with edited chunk of samples.
        """
        data = input_data.copy()
        data[self.text_col] = data[self.text_col].apply(lambda row: self.stopword_remove(row))
        return data

    def flag_lang(self, input_data:pd.DataFrame) -> pd.DataFrame:
        """Detect Language of each sample (row) containing text of one crawled website. 

        Args:
            input_data (pd.DataFrame): DataFrame containing chunk of samples.

        Returns:
            pd.DataFrame: DataFrame with edited chunk of samples. 
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
        
        data = input_data.copy()
        data["LANG"]= data[self.text_col].apply(lambda row: detect_language(row))
        if self.lang != None:
            data = data[data["LANG"] == self.lang]
        data = data.reset_index(drop = True)
        return data

    def lemmatize_text(self, input_data:pd.DataFrame) -> pd.DataFrame:
        """Lemmatize text with help of spacy

        Args:
            input_data (pd.DataFrame): DataFrame containing chunk of samples.

        Returns:
            pd.DataFrame: DataFrame with edited chunk of samples. 
        """
        data = input_data.copy()
        data[self.text_col] = data[self.text_col].apply(lambda row: " ".join([token.lemma_ for token in nlp(row)]))
        return data

    def remove_cityNames(self, input_data:pd.DataFrame) -> pd.DataFrame:
        """Removes all city names in text

        Args:
            input_data (pd.DataFrame): DataFrame containing chunk of samples.

        Returns:
            pd.DataFrame: DataFrame with edited chunk of samples.
        """
        data = input_data.copy()
        regex = re.compile("|".join(map(re.escape, self.cities.keys(  ))))
        data[self.text_col] = data[self.text_col].apply(lambda row: regex.sub(lambda match: self.cities[match.group(0)], row) if row else "")
        return data
    
    def split_dataframe(self, chunk_size:int = 300) -> list:
        """Helper function that splits loaded dataset into smaller chunks containing size "chunk_size" which is by default 300 samples.

        Args:
            chunk_size (int, optional): Size of DataFrame chunk. Defaults to 300.

        Returns:
            list: Returns a list of DataFrames each containting a sampleset of 300 samples. All DataFrames in list result in the total dataset.
        """
        chunks = list()
        num_chunks = math.ceil(len(self.data) / chunk_size)
        for i in range(num_chunks):
            chunks.append(self.data[i*chunk_size:(i+1)*chunk_size])
        return chunks

    def save_data(self, cleaned_chunk:pd.DataFrame):
        """Concatenate new chunk of cleaned data to already exisiting cleaned data.

        Args:
            cleaned_chunk (pd.DataFrame): Chunk of 300 (by default) cleaned samples of data.
        """
        df_t_path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(df_t_path):
            cleaned_data = pd.read_feather(df_t_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        else:
            cleaned_data = pd.DataFrame(columns=cleaned_chunk.columns.tolist())

        data_to_save = pd.concat([cleaned_data,cleaned_chunk], ignore_index=True).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True) 
        print(data_to_save.shape)

        data_to_save.to_feather(df_t_path)

    def run(self):
        """Run function of textFilter class. Firstly a basic text cleansing will be applied, than an advanced text cleaning and advanced stopwords removal will be applied.
        Than the language of cleaned texts is applied and all texts that are not matching the initialised lang are filtered. In advance all city names in texts 
        will be removed and text will be lemmatized. Filtered and cleaned texts are finally saved to target path. 
        """
        df_chunks = self.split_dataframe()
        print("Size of full dataset: {dataset}. Number of chunks: {chunks}".format(dataset = self.data.shape[0], chunks = len(df_chunks)))
        logging.info("[{log}]Data cleaning started".format(log = datetime.now()))
        for i,chunk in enumerate(df_chunks):
            try:
                print("New chunk starts cleaning")
                chunk_text = self.remove_nonText(chunk)
                chunk_stopwords = self.remove_domainStopwords(chunk_text)
                print("Non textual elements and stopwords had been removed.")
                chunk_lang = self.flag_lang(chunk_stopwords)
                print("Languages had been detected and filtered.")
                chunk_lem = self.lemmatize_text(chunk_lang)
                print("Text had been lemmatized.")
                chunk_cit = self.remove_cityNames(chunk_lem)
                print("City names had been removed.")
                self.save_data(chunk_cit)
                print("[{log}]Data chunk {number} with {size} of {shape} total samples had been cleaned.".format(number = i,size =chunk.shape, shape =self.data.shape[0], log = datetime.now()))
                logging.info("[{log}]Data chunk {number} with {size} of {shape} total samples had been cleaned.".format(number = i,size =chunk.shape, shape = self.data.shape[0], log = datetime.now()))
            except KeyboardInterrupt:
                print(KeyboardInterrupt)
                logging.warning('[{log}]Data cleaning of a chunk of samples had been interrupted by KeyboardInterrupt.'.format(log = datetime.now()))
                return
            except Exception as e:
                print(e)
                logging.warning('[{log}]Something with data cleaning of a chunk of samples went wrong: {error}.'.format(error =e, log = datetime.now() ))
                return
                
            
        

if __name__ == "__main__":
    d = textFilter('de',r"files\raw_texts.feather",r"files\cleaned_texts.feather")
    d.run()
    # e = textFilter('en',r"files\raw_texts_en.feather",r"files\cleaned_texts_en.feather")
    # e.run()


    # data_sample = f.data.sample(frac = 0.007,replace = False,random_state = 1, axis = 0)
    # print(data_sample.shape)
    # print(set(data_sample['CLASS'].tolist()))
    #f.run()
    # f2 = textFilter("de",r"files\raw_classes.feather",r"files\cleaned_classes.feather")
    # f2.run()
    #re.sub( "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.(de|com)\\b(?:[-a-zäöüßA-Z0-9()@:%_\\+.~#?&\\/=]*)$", "", w)
    # result = [re.search( "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.(de|com)\\b(?:[-a-zäöüßA-Z0-9()@:%_\\+.~#?&\\/=]*)$",w) for w in ["https://www.abarth.fr", "https://www.abarth.de","https://www.abarth.gr", "https://www.abarth.com"]]
    # print(result)