from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import os
import nltk
import pickle
nltk.download('stopwords')

class TOPIC_KMeans:

    def __init__(self, topics_path:str,data_path:str):
        self.topics = self.load_centroids(topics_path)
        self.raw_data = self.load_data(data_path)

    def load_centroids(self,source_path):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
        return pd.read_feather(df_path)

    def load_data(self,source_path):
        df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
        return pd.read_feather(df_path)

    def generate_TfIdf(self,text):
        tokenizer = RegexpTokenizer(r'\w+')
        german_stopwords = stopwords.words('german')

        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=german_stopwords,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        tfidf_v = tfidf_v.fit(text)
        return tfidf_v

    def apply_kMeans(self,data,centroids):
        kmeans = KMeans(init = centroids, n_clusters=7, random_state=1, n_init = 1)
        kmeans.fit(centroids)
        print(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        print(cluster_centers)

        self.save_model(kmeans)
        

    def save_model(self,model):
        with open("kmeans.pkl", "wb") as f:
            pickle.dump(model, f)

    def run(self):
        dic = {}
        for i, t in enumerate (self.topics['TOPIC'].tolist()):  #replace with CLASS!
            print(i,t)
            dic[i] = t

        raw_centroids = self.topics['TOPICS'].tolist()
        all_texts = raw_centroids + self.raw_data['TOPIC'].tolist()

        tfidf = self.generate_TfIdf(all_texts)
        text = tfidf.transform(self.raw_data['TOPIC'].tolist())
        centroids = tfidf.transform(raw_centroids)
        print(centroids.shape)

        self.apply_kMeans(text.toarray(),centroids.toarray())


kmeans = TOPIC_KMeans(r"files\topiced_topics.feather",r"files\topiced_texts.feather")

# return kmeans.predict(data)
# raw_data['CLUSTER'] = apply_kMeans(text.toarray(),centroids.toarray())
# raw_data['CLUSTER'] = raw_data['CLUSTER'].map(dic)
# print(raw_data)

