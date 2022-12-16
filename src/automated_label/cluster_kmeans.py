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
    """Class to predict predefined clusters (topics/classes) with kMeans Algorithm based on predefined fixed centroids.
    The centroids are based on TOPIC_Classes.xlsx with cleaned and generated topics in topiced_classes.feather 
    """
    def __init__(self, topics_path:str,data_path:str):
        """Initialisation of a cluster generator object. Loads predefined clusters and cleaned texts to predict clusters on.

        Args:
            topics_path (str): Path to predefined cluster data.
            data_path (str): Source path to file containing cleaned texts and generated topics to predict cluster.
        """
        self.topics = self.load_centroids(topics_path)
        self.raw_data = self.load_data(data_path)

    def load_centroids(self,source_path:str) -> pd.DataFrame:
        """Load data containing predefined clusters. The data contain the Cluster (CLASS), the url link (DOMAIN), 
        the url link (URL), the cleaned texts(URL_TEXT), the language of text (LANG) and the identfied topics (TOPIC).
        The CLASSES with related TOPICs form the fix centroids of the fitted model.

        Args:
            source_path (str): Source path to file containing topics to generate fix centrodis.

        Returns:
            pd.DataFrame: Returns Returns a pandas DataFrame containing cluster names (CLASS), domain name of url link (DOMAIN), url link (URL), cleaned texts(URL_TEXT), language of text (LANG) and 
            generated topics (TOPIC) based on cleaned texts. 
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
        return pd.read_feather(df_path)

    def load_data(self,source_path:str) -> pd.DataFrame:
        """Load data containing data for which clusters are to be predicted. The data contain the domain of url link (DOMAIN), 
        the url link (URL), the cleaned texts(URL_TEXT), the language of text (LANG) and the identfied topics (TOPIC).
        The cluster will be predicted based on the TOPICs.

        Args:
            source_path (str): Source path to file containing topics to predict cluster.

        Returns:
            pd.DataFrame: Returns a pandas DataFrame containing domain name of url link (DOMAIN), url link (URL), cleaned texts(URL_TEXT), language of text (LANG) and 
            generated topics (TOPIC) based on cleaned texts. 
        """
        df_path = str(os.path.dirname(__file__)).split("src")[0] + source_path
        return pd.read_feather(df_path)

    def generate_TfIdf(self,text:list):
        """Generation of Tfidf-Vector and fit to all topics (TOPIC) which the clusters are to be predicted and the topics (TOPIC) of the data containing predefined clusters.

        Args:
            text (list): List of strings of all topics of data for which clusters are to be predicted and all topics of data containing predefined clusters.

        Returns:
            TfidfVectorizer: Returns Tfifd Vectorizer fitted to all topics of data for which clusters are to be predicted and all topics of data containing predefined clusters.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        german_stopwords = stopwords.words('german')

        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=german_stopwords,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        tfidf_v = tfidf_v.fit(text)
        return tfidf_v

    def apply_kMeans(self,centroids: np.array):
        """Generation and fit of KMeans Model and saving of fitted Model.

        Args:
            centroids (np.array): Fix defined centroids based on TOPICs of data containing predefined clusters.
        """
        kmeans = KMeans(init = centroids, n_clusters=7, random_state=1, n_init = 1)
        kmeans.fit(centroids)
        print(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        print(cluster_centers)

        self.save_model(kmeans)
        

    def save_model(self,model:KMeans):
        """Saving of given KMeans model as pickle file.

        Args:
            model (KMeans): Fitted KMeans model.
        """
        path = str(os.path.dirname(__file__)).split("src")[0] + "models\kmeans.pkl"
        if os.path.exists(path):
            os.remove(path)
        with open("models\kmeans.pkl", "wb") as f:
            pickle.dump(model, f)

    def save_clusterNames(self):
        dic = {}
        for i, t in enumerate (self.topics['CLASS'].tolist()): 
            print(i+1,t)
            dic[str(i+1)] = [t]
        #save cluster dictionary
        cluster_names =  pd.DataFrame.from_dict(dic)
        cluster_names.to_feather(r"files\kMeans_cluster.feather")

    def run(self):
        """Run function of TOPIC_KMeans class. After data and centroids are seperately loaded the topics of the data to be predicted and the topics of the predefined clusters 
        are merged. This text corpus is used to fit and transform a KMeans model which is than saved.
        """
        raw_centroids = self.topics['TOPIC'].tolist()
        all_texts = raw_centroids + self.raw_data['TOPIC'].tolist()

        tfidf = self.generate_TfIdf(all_texts)
        text = tfidf.transform(self.raw_data['TOPIC'].tolist())
        centroids = tfidf.transform(raw_centroids)
        print(centroids.shape)

        self.apply_kMeans(centroids.toarray())
        self.save_clusterNames()


kmeans = TOPIC_KMeans(r"files\topiced_classes.feather",r"files\topiced_texts.feather")
kmeans.run()

