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
from lda_coherence import get_coherence 
import numpy as np
import nltk
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
       

    def load_data(self):
        """Read cleaned text stored in source path

        Returns:
            DataFrame: Returns a pandas DataFrame containing domain name of url link (DOMAIN), url link (URL), cleaned texts(URL_TEXT), language of text (LANG) and CLASS (optional).
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        if self.topic:
            data = pd.read_feather(df_path).groupby("CLASS").agg({'URL_TEXT':lambda x: "|".join(list(x))})#['URL_TEXT'].apply(list)
        else:
            data = pd.read_feather(df_path)
        #group by class#
        return data

    def save_data(self):
        """Save data as feather file to defined target path.
        """
        self.data = self.data.reset_index()
        path = str(os.path.dirname(__file__)).split("src")[0] + self.target_path
        if os.path.exists(path):
            os.remove(path)
        self.data.to_feather(path)

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


    def run(self):
        """Run function of TopicExtractor class. First generate_topic() is rowwise called, than empty values will be replaced by empty strings.
        """
        self.data['TOPIC']=self.data[self.text_col].apply(lambda row: self.generate_topic(row))
        self.data.replace(np.nan, "",regex = False, inplace = True)
        self.save_data()


if __name__ == "__main__":
    t = TopicExtractor(7,r"files\cleaned_texts.feather",r"files\topiced_texts.feather", "de")
    t.run() 
    # print(t.data['TOPIC'].tolist()[:3])
    t2 = TopicExtractor(7,r"files\cleaned_classes.feather",r"files\topiced_classes.feather","de",True)
    t2.run()
    # print(t2.data['TOPIC'].tolist())