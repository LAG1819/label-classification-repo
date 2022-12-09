from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import os
import nltk
nltk.download('stopwords')

def load_centroids(source_path):
    df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
    return pd.read_feather(df_path)

def load_data(source_path):
    df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
    return pd.read_feather(df_path)

def generate_TfIdf(text):
    tokenizer = RegexpTokenizer(r'\w+')
    german_stopwords = stopwords.words('german')

    tfidf_v = TfidfVectorizer(lowercase=True,
                            stop_words=german_stopwords,
                            ngram_range = (1,2),
                            tokenizer = tokenizer.tokenize)

    tfidf_v = tfidf_v.fit(text)
    return tfidf_v

def apply_kMeans(data,centroids):
    kmeans = KMeans(init = centroids, n_clusters=7, random_state=1, n_init = 1)
    kmeans.fit(centroids)
    print(kmeans.labels_)
    cluster_centers = kmeans.cluster_centers_
    return kmeans.predict(data)

topics = load_centroids(r"files\topiced_topics.feather")
raw_data = load_data(r"files\topiced_texts.feather")
dic = {}
for i, t in enumerate (topics['TOPIC'].tolist()): 
    print(i,t)
    dic[i] = t

raw_centroids = topics['TOPICS'].tolist()
all_texts = raw_centroids + raw_data['TOPIC'].tolist()

tfidf = generate_TfIdf(all_texts)
text = tfidf.transform(raw_data['TOPIC'].tolist())
centroids = tfidf.transform(raw_centroids)
print(centroids.shape)

raw_data['CLUSTER'] = apply_kMeans(text.toarray(),centroids.toarray())
raw_data['CLUSTER'] = raw_data['CLUSTER'].map(dic)
print(raw_data)