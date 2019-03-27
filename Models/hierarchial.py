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


class Hierarchial(Model):
    def __init__(self, num_clusters, feature_names, train_x, train_y):
        self.hierarchial_model = AgglomerativeClustering(n_clusters=num_clusters).fit(train_x)
        self.feature_names = feature_names
        self.train_x = train_x
        self.train_y = train_y
        self.labels = self.hierarchial_model.labels_
        self.num_clusters = num_clusters
        self.sil_score = -100.0
        self.db_idx_score = -100.0

    def _tokenize(self, query):
        regex_string = r'[a-zA-Z]+'
        tokenizer = RegexpTokenizer(regex_string)
        tokens = tokenizer.tokenize(query.lower())
        tokens = [x for x in tokens if x.isalpha()]
        tokens = [x for x in tokens if len(x) > 2]
        return tokens

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = self._tokenize(query)

        # get representation (bag of words)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        pred_y = self.hierarchial_model.predict(weights)

        return pred_y

    def get_nearest_neighbours(self, query):
        # get cluster id
        pred_y = self._get_cluster_id(query)

        # tokenize data and get bag of words
        tokens = self._tokenize(query)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        print(weights)

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
        for i in range(5):
            print(sorted_by_dist[i])
            print(np.where(sorted_by_dist[i].toarray() == 1)[1])
            for x in np.where(cluster_set[i].toarray() != 0)[1]:
                print(self.feature_names[x])

    def eval(self):
        self.sil_score = metrics.silhouette_score(self.train_x, self.labels, metric='euclidean')
        self.db_idx_score = metrics.davies_bouldin_score(self.train_x, self.labels)
        #         print(self.sil_score)
        #         print(self.db_idx_score)

        # evaluate with ON WG IDENTIFIER
        self.custom_score()

    def custom_score(self):
        # evaluate with ON WG IDENTIFIER
        # how to get the number of ids
        # for each cluster count the number onwgid occurences
        # get unique list of cluster ids

        #         unique_cids = self.train_y['ON WG IDENTIFIER'].unique()
        unique_cids = np.unique(self.train_y)
        print(len(unique_cids))

    #         print(y)
    # for cid in unique_cids:
    # cluster: (onwgid1: 1, onwgid2)

    def get_sil_score(self):
        return metrics.silhouette_score(self.train_x, self.labels)

    def get_db_idx_score(self):
        return davies_bouldin_score(self.train_x, self.labels)

    def get_labels(self):
        return self.labels
