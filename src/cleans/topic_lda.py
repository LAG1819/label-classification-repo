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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
try:
    from src.cleans.lda_coherence import get_coherence 
except:
    from lda_coherence import get_coherence
import numpy as np
import nltk
import math
import logging
from datetime import datetime
nltk.download('stopwords')

class TopicExtractor:
    """Class to identify topics of cleand texts (cleaned_texts.feather) based on LDA(LatentDirichletAllocation) algorithm. 
    """
    def __init__(self, input_topic: int, s_path:str,t_path:str, lang:str, topic:bool = False):
        """Initialisation of a topic generator object. 

        Args:
            input_topic (int): Number of topics, that shall be generated
            s_path (str): source path to file containing cleaned texts to identify topics
            t_path (str): target path to save file with cleaned texts and generated topics
            lang (str): unicode of language to filter raw texts only in that language 
        """
        self.source_path = s_path
        self.target_path = t_path
        self.topic = topic
        self.data = self.load_data()
        self.number_topics = input_topic
        self.text_col = 'URL_TEXT'
        self.lang = lang

        if self.lang == "de":
            self.stopwords = stopwords.words('german')
        elif self.lang == 'en':
            self.stopwords = stopwords.words('english')
        else:
            self.stopwords = stopwords.words('german')

        filenames =  str(os.path.dirname(__file__)).split("src")[0] +r'doc\topic_extraction_'+lang+'.log'
        logging.basicConfig(filename=filenames, encoding='utf-8', level=logging.DEBUG)
        logging.info("Topic Extraction with Language {l} and data file {path} (source) created. Target file is {tpath}".format(l = self.lang, path = self.source_path, tpath = self.target_path))
       

    def load_data(self):
        """Read cleaned text stored in source path

        Returns:
            DataFrame: Returns a pandas DataFrame containing domain name of url link (DOMAIN), url link (URL), cleaned texts(URL_TEXT), language of text (LANG) and CLASS (optional).
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        df_t_path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path

        #if topic dataset than group by class#
        if self.topic:
            data = pd.read_feather(df_path)
            raw_data = data.groupby("CLASS").agg({'URL_TEXT':lambda x: "|".join(list(x))})#['URL_TEXT'].apply(list)
            logging.info("[{log}]Total of k-Means centroid data to extract topics: {all}. Total of data with no extracted topics yet:{l}".format(log=datetime.now(), all = data.shape, l=raw_data.shape))
            print("[{log}]Total of k-Means centroid data to extract topics: {all}. Total of data with no extracted topics yet:{l}".format(log=datetime.now(), all = data.shape, l=raw_data.shape))
        else:
            data = pd.read_feather(df_path)

            if os.path.exists(df_t_path):
                topiced_data = pd.read_feather(df_t_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
            else:
                topiced_data = pd.DataFrame(columns=data.columns.tolist())          

            # Identify what values are in cleaned data and not already in topiced data
            left_join = data.merge(topiced_data, on='URL', how='left', indicator=True)
            left_join_df = left_join.loc[left_join['_merge'] == 'left_only', 'URL']
            raw_data = data[data['URL'].isin(left_join_df)]
            if "TOPIC" in raw_data.columns:
                raw_data = raw_data.drop(columns="LANG", axis = 1)  

            print("Raw data (total):", data.shape)
            print("Raw data:", raw_data.shape)
            print("Topiced data:", topiced_data.shape)
            logging.info("[{log}]Total of data to extract topics: {all}. Total of data already with extracted topic: {t}. Total of data with no extracted topics yet:{l}".format(log=datetime.now(), all = data.shape, t = topiced_data.shape,l=raw_data.shape))
            print("[{log}]Total of data to extract topics: {all}. Total of data already with extracted topic: {t}. Total of data with no extracted topics yet:{l}".format(log=datetime.now(), all = data.shape, t = topiced_data.shape,l=raw_data.shape))          
        return raw_data


    def generate_tfIdf(self,doc_list:list):
        """Apply rowwise generation of Tfidf-Vector and fit to cleaned texts (URL_TEXT).

        Args:
            doc_list (list): Rowwise list of cleaned texts (URL_TEXT) of dataset.

        Returns:
            tfidf_matrix, idf np_array: Returns transformed cleaned texts to tfidf matrix and generated Tfif Vector based on doc_list. 
        """
        tokenizer = RegexpTokenizer(r'\w+')
        
        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=self.stopwords,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        fit_data = tfidf_v.fit_transform(doc_list)
        
        return fit_data,tfidf_v   

    def apply_lda(self,fit_data,tfidf_v) -> str:
        """Fit and applies LDA Algorithm on tfidf matrix of cleaned texts rowwise and returns a list of generated topics.

        Args:
            fit_data (tfidf_matrix): rowwise tfidf matrix of transformed cleaned texts.
            tfidf_v (idf np_array): rowwise generated Tfif Vector.

        Returns:
            str: Returns generated topics of rowwise cleaned texts.
        """

        model=LatentDirichletAllocation(n_components=self.number_topics)

        lda_dtm = model.fit(fit_data)
        lda_matrix = model.fit_transform(fit_data)

        lda_components=model.components_

        terms = tfidf_v.get_feature_names_out()
        
        topics = []
        for index, component in enumerate(lda_components):
            zipped = zip(terms, component)
            top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
            top_terms_list=list(dict(top_terms_key).keys())
            topics.append(top_terms_list)

        topics = [item for sublist in topics for item in sublist]
        topics = list(set(list(filter(lambda x: len(x) > 3, topics))))

        ##Test Coherence Values##
        # tfiIdf_vectorizer_vocab = np.array([x for x in tfidf_v.vocabulary_.keys()])
        # c = get_coherence(lda_components,lda_dtm,fit_data,tfiIdf_vectorizer_vocab)
        # print(c)

        return ",".join(topics)

    def generate_topic(self,text:str) -> str:
        """Generation, Fit and Application of cleaned text rowwise by calling generate_tfIdf and apply_lda function.  

        Args:
            text (str): one sample (row) containing cleaned text of one crawled website.

        Returns:
            str: String list of generated topics of one sample.
        """
        try:
            doc_list = text.split(" ")
            fitted_data, tfidf = self.generate_tfIdf(doc_list)
            topics = self.apply_lda(fitted_data,tfidf)
            # print(topics.split("|"))
            return topics
        except Exception as e:
            print(e)
            return ''

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

    def save_data(self, topiced_chunk:pd.DataFrame):
        """Concatenate new chunk of generated topics data to already exisiting data containing topics.

        Args:
            topiced_chunk (pd.Dataframe): Chunk of 300 (by default) samples of data with extracted topics.
        """
        df_t_path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if self.topic:            
            data_to_save = topiced_chunk
            data_to_save.reset_index(inplace=True)
        else:
            if os.path.exists(df_t_path):
                topiced_data = pd.read_feather(df_t_path).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
            else:
                topiced_data = pd.DataFrame(columns=topiced_chunk.columns.tolist())

            data_to_save = pd.concat([topiced_data,topiced_chunk], ignore_index=True).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True) 
        
        print(data_to_save.shape)
        print("Topic extraction of one data sampleset finished.")
        data_to_save.to_feather(df_t_path)


    def run(self):
        """Run function of TopicExtractor class. First generate_topic() is rowwise called, than empty values will be replaced by empty strings.
        """
        df_chunks = self.split_dataframe()
        print("Size of full dataset: {dataset}. Number of chunks: {chunks}".format(dataset = self.data.shape[0], chunks = len(df_chunks)))
        logging.info("[{log}]Topic extraction started".format(log = datetime.now()))
        for i, chunk in enumerate(df_chunks):
            logging.info("[{log}]Topic extraction with LDA of data sampleset {number} with size {size} started".format(log = datetime.now(), number = i, size = chunk.shape))
            chunk_c = chunk.copy()
            chunk_c['TOPIC']=chunk_c[self.text_col].apply(lambda row: self.generate_topic(row))
            chunk_c.replace(np.nan, "",regex = False, inplace = True)
            self.save_data(chunk_c)
            logging.info("[{log}]Topic extraction of data sampleset {number} with size {size} of dataframe {df} is finished.".format(log = datetime.now(), number = i,size =chunk_c.shape, df = self.data.shape))
        logging.info("[{log}]Topic extraction of dataframe {df} is finished.".format(log = datetime.now(), df = self.data.shape))

# if __name__ == "__main__":
#     lang = 'de'
#     texts_topics = TopicExtractor(2,r"files\02_clean\cleaned_texts_"+lang+r".feather",r"files\02_clean\topiced_texts_"+lang+r".feather", lang)
#     texts_topics.run() 
#     class_topics = TopicExtractor(2,r"files\02_clean\cleaned_classes_"+lang+r".feather",r"files\02_clean\topiced_classes_"+lang+r".feather",lang,True)
#     class_topics.run()
    
