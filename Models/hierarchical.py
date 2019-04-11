from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib as MPL

MPL.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import sys, os, csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Models.model import Model
from Models.clustermodel import ClusterModel
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pickle
import seaborn as sns
from data_manipulator import *
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings

warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score


class Hierarchical(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.hierarchical_model = AgglomerativeClustering(n_clusters=num_clusters).fit(train_x)
        self.labels = self.hierarchical_model.labels_
        self.num_clusters = num_clusters

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = self._tokenize(query)
        print(type(self.train_x))

        # get representation (bag of words)
        pred_y =""
        if self.vectorizer == "CountVectorizer":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
            pred_y = self.hierarchial_model.predict(weights)
        elif self.vectorizer == "TfidfVectorizer":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
            pred_y = self.hierarchial_model.predict(weights)
        return pred_y

    def get_nearest_neighbours(self, query):
        # get cluster id
        pred_y = self._get_cluster_id(query)

        # tokenize data and get bag of words
        #tokens = self._tokenize(query)
        #weights, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
        #print(weights)

        # get all training values in same cluster
        train_x_idx = np.where(self.labels == pred_y)
        cluster_set = self.train_x[np.where(self.labels == pred_y)]
        print("Clustered with: " + str(cluster_set.shape[0]))

        # get distances between query and all in same cluster
        distances = [euclidean_distances(weights, train_example.reshape(1, -1)) for train_example in cluster_set]

        # sort by distance
        sorted_dist = [d for d, x in sorted(zip(distances, cluster_set), key=lambda x: x[0])]
        sorted_by_dist = [x for d, x in sorted(zip(distances, cluster_set), key=lambda x: x[0])]

        # print the closest neighbours
        for i in range(2):
            print(sorted_by_dist[i])
            print(np.where(sorted_by_dist[i].toarray() == 1)[1])
            for x in np.where(cluster_set[i].toarray() != 0)[1]:
                print(self.feature_names[x])
