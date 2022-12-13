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
    def __init__(self, input_topic, s_path,t_path):
        self.source_path = s_path
        self.target_path = t_path
        self.data = self.load_data()
        self.number_topics = input_topic
        self.text_col = 'URL_TEXT'
        self.german_stopwords = stopwords.words('german')
       

    def load_data(self):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + self.source_path
        #group by class#
        return pd.read_feather(df_path)

    def save_data(self):
        self.data.to_feather(str(os.path.dirname(__file__)).split("src")[0] + self.target_path)
        

    def generate_tfIdf(self,doc_list):
        tokenizer = RegexpTokenizer(r'\w+')
        
        city_path = str(os.path.dirname(__file__)).split("src")[0] + r"files/germany.json"
        german_cites = pd.read_json(city_path)["name"].tolist()
        german_cites = [c.lower() for c in german_cites]
        
        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=self.german_stopwords + german_cites,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        fit_data = tfidf_v.fit_transform(doc_list)
        
        return fit_data,tfidf_v   

    def apply_lda(self,fit_data,tfidf_v):

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

        return "|".join(topics)

    def generate_topic(self,text):
        try:
            doc_list = text.split("|")
            fitted_data, tfidf = self.generate_tfIdf(doc_list)
            topics = self.apply_lda(fitted_data,tfidf)
            # print(topics.split("|"))
            return topics
        except Exception as e:
            print(e)
            return ''


    def run(self):
        self.data['TOPIC']=self.data[self.text_col].apply(lambda row: self.generate_topic(row))
        self.data.replace(np.nan, "",regex = False, inplace = True)
        # self.save_data()


if __name__ == "__main__":
    t = TopicExtractor(7,r"files\cleaned_texts.feather",r"files\topiced_texts.feather")
    t.run() 
    print(t.data['TOPIC'].tolist())
    # t2 = TopicExtractor(7,r"files\cleaned_classes.feather",r"files\topiced_classes.feather")
    # t2.run()
    # print(t2.data['TOPIC'].tolist())