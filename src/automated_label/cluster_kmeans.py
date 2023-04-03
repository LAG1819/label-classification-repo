# <Pretraining a k-Means Cluster based of user-defined classes for Automated Labeling. This is a optional preprocessing for TOTAL data labeling for step 4.>
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

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import os
import nltk
import pickle
nltk.download('stopwords', quiet=True)

class TOPIC_KMeans:
    """Class to predict predefined clusters (topics/classes) with kMeans Algorithm based on predefined fixed centroids.
    The centroids are based on TOPIC_Classes.xlsx with cleaned and generated topics in topiced_classes.feather 
    """
    def __init__(self,lang:str ,topics_path:str= None,data_path:str=None, nbr_clusters:int = 7):
        """Initialisation of a cluster generator object. Loads predefined clusters and cleaned texts to predict clusters on.
         
         Args:
            lang (str): Unicode of language selected texts to create cluster with.
            topics_path (str, optional): Path to predefined cluster data. Defaults to None.
            data_path (str, optional):  Source path to file containing cleaned texts and generated topics to predict cluster. Defaults to None.
            nbr_clusters (int, optional): Numbers of cluster which is equivalent to the number of classes for automated labeling. Defaults to 7.
        """
        self.__lang = lang
        self.__number_clusters = nbr_clusters
        if topics_path:
            self.__topics = self.__load_centroids(topics_path)
        else:
            self.__topics = self.__load_centroids(r"files\topiced_classes_"+self.__lang+r".feather")
        if data_path:
            self.__raw_data = self.__load_data(data_path)
        else:
            self.__raw_data = self.__load_data(r"files\topiced_texts_"+self.__lang+r".feather")
        
    def __load_centroids(self,source_path:str) -> pd.DataFrame:
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
        df = pd.read_feather(df_path)
        df.reset_index(drop=True, inplace=True)
        return df

    def __load_data(self,source_path:str) -> pd.DataFrame:
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

    def __generate_TfIdf(self,text:list):
        """Generation of Tfidf-Vector and fit to all topics (TOPIC) which the clusters are to be predicted and the topics (TOPIC) of the data containing predefined clusters.

        Args:
            text (list): List of strings of all topics of data for which clusters are to be predicted and all topics of data containing predefined clusters.

        Returns:
            TfidfVectorizer: Returns Tfifd Vectorizer fitted to all topics of data for which clusters are to be predicted and all topics of data containing predefined clusters.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        if self.__lang == 'de':
            stopwords_l = stopwords.words('german')
        if self.__lang == 'en':
            stopwords_l = stopwords.words('english')

        tfidf_v = TfidfVectorizer(lowercase=True,
                                stop_words=stopwords_l,
                                ngram_range = (1,2),
                                tokenizer = tokenizer.tokenize)

        tfidf_v = tfidf_v.fit(text)

        return tfidf_v

    def __apply_kMeans(self,centroids: np.array, vectorizer:TfidfVectorizer):
        """Generation and fit of KMeans Model and saving of fitted Model.

        Args:
            centroids (np.array): Fix defined centroids based on TOPICs of data containing predefined clusters.
        """
        kmeans = KMeans(init = centroids, n_clusters=self.__number_clusters, random_state=1, n_init = 1)
        kmeans.fit(centroids)
        #print(kmeans.labels_)
        cluster_centers = kmeans.cluster_centers_
        # print(cluster_centers)

        self.__save_model(kmeans,vectorizer)
        

    def __save_model(self,model:KMeans, vectorizer:TfidfVectorizer):
        """Saving of given KMeans model as pickle file and the fitted tfidf vectorizer.

        Args:
            model (KMeans): Fitted KMeans model.
        """
        path_m = str(os.path.dirname(__file__)).split("src")[0] + "models\label\k_Means\kmeans_"+str(self.__lang)+".pkl"
        path_v = str(os.path.dirname(__file__)).split("src")[0] + "models\label\k_Means\kmeans_vectorizer_"+str(self.__lang)+".pkl"


        if os.path.exists(path_m):
            os.remove(path_m)
        if os.path.exists(path_v):
            os.remove(path_v)

        with open(path_m, "wb") as f:
            pickle.dump(model, f)

        with open(path_v, 'wb') as fin:
            pickle.dump(vectorizer, fin)

    def __save_clusterNames(self):
        """The function stores the different clusters and the corresponding cluster names in a feather file.
        This function is a helper function if it is necessary to evaluate whether the clusters and their names match the classes and names in Automated Labeling.
        """
        dic = {}
        for i, t in enumerate (self.__topics['CLASS'].tolist()): 
            print(i,t)
            dic[str(i)] = [t]
        #save cluster dictionary
        _cluster_names =  pd.DataFrame.from_dict(dic)

        _path = str(os.path.dirname(__file__)).split("src")[0] +r"files\03_label\k_Means\kMeans_cluster_"+str(self.__lang)+".feather"
        if not os.path.exists(_path):
            os.makedirs(_path)
        _cluster_names.to_feather(_path)

    def run(self):
        """Run function of TOPIC_KMeans class. After data and centroids are seperately loaded the topics of the data to be predicted and the topics of the predefined clusters 
        are merged. This text corpus is used to fit and transform a KMeans model which is than saved.
        """
        #load centroids and dataset and concatenate to one large text document
        raw_centroids = self.__topics['TOPIC'].tolist()
        all_texts = raw_centroids + self.__raw_data['TOPIC'].tolist()

        #generate TF-IDF Vectorizer and centroids based on generated large text document
        tfidf = self.__generate_TfIdf(all_texts)
        text = tfidf.transform(self.__raw_data['TOPIC'].tolist())
        centroids = tfidf.transform(raw_centroids)

        #generate preset k-means cluster and save it as well as TF-IDF Vectorizer
        self.__apply_kMeans(centroids.toarray(), tfidf)
        self.__save_clusterNames()