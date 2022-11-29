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
    def __init__(self, input_topic:int):
        self.data = self.load_data()
        self.number_topics = input_topic
        self.text_col = 'URL-TEXT'
        self.german_stopwords = stopwords.words('german')

    def load_data(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_pre_clean.csv"
        return pd.read_csv(df_path, header = 0, delimiter=",")

    def save_data(self):
        self.data.to_csv(str(os.path.dirname(__file__)).split("src")[0] + r"files\Output_texts_cleaned.csv", index = False)
        

    def generate_tfIdf(self,doc_list):
        tokenizer = RegexpTokenizer(r'\w+')

        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=self.german_stopwords,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        fit_data = tfidf_v.fit_transform(doc_list)
        
        return fit_data,tfidf_v   

    def apply_lda(self,fit_data,tfidf_v):

        model=LatentDirichletAllocation(n_components=self.number_topics)

        lda_dtm = model.fit(fit_data)
        lda_matrix = model.fit_transform(fit_data)

        lda_components=model.components_

        terms = tfidf_v.get_feature_names()
        
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

        return "|".join(topics)

    def generate_topic(self,text):
        try:
            doc_list = text.split("|")
            fit, tfidf = self.generate_tfIdf(doc_list)
            topics = self.apply_lda(fit,tfidf)
            # print(topics.split("|"))
            return topics
        except:
            return ''


    def run(self):
        self.data['TOPIC']=self.data[self.text_col].apply(lambda row: self.generate_topic(row))
        self.data.replace(np.nan, "",regex = False, inplace = True)
        self.save_data()


if __name__ == "__main__":
    t = TopicExtractor(4)
    t.run() 
    # print(t.data['TOPIC'].tolist())