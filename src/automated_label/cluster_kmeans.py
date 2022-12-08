from sklearn.cluster import KMeans
import numpy as np

def load_centroids():
    pass

def load_data():
    pass

centroids = load_centroids()
data = load_data()
kmeans = KMeans(init = centroids, n_clusters=7, random_state=1, n_init = 1)
kmeans.fit(centroids, init=centroids, n_init=1)
kmeans.cluster_centers_
kmeans.predict(data)