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
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
import math
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))

class ClusterModel(Model):
    def __init__(self, train_x, train_y, feature_names, rep):
        self.train_x = train_x
        self.train_y = train_y
        self.labels = list()
        self.feature_names = feature_names
        self.rep = rep
        self.sil_score = -100.0
        self.db_idx_score = -100.0

    def _tokenize(self, query):
        regex_string = r'[a-zA-Z]+'
        tokenizer = RegexpTokenizer(regex_string)
        tokens = tokenizer.tokenize(query.lower())
        tokens = [x for x in tokens if x.isalpha()]
        tokens = [x for x in tokens if len(x) > 1]
        return tokens

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = self._tokenize(query)
        print(type(self.train_x))

        # get representation (bag of words)
        pred_y =""
        if self.rep == "bow":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
            pred_y = self.hierarchial_model.predict(weights)
        elif self.rep == "tfidf":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
            pred_y = self.hierarchial_model.predict(weights)
        return pred_y

    def eval(self):
        self.sil_score = metrics.silhouette_score(self.train_x, self.labels, metric='euclidean')
        self.db_idx_score = metrics.davies_bouldin_score(self.train_x, self.labels)

    def custom_score(self):
        # evaluate with ON WG IDENTIFIER
        unique_cids = np.unique(self.train_y)
        print(len(unique_cids))
        # for each cluster
        scores = list() 
        for i in set(self.labels.tolist()):
            print("cluster: " + str(i))

            # get ON WG IDENTIFIERS for all in same cluster
            cluster_cids_i = list()
            for j in np.where(self.labels == i)[0]:
                #print(self.train_y.iloc[j].values)
                cluster_cids_i.append(self.train_y.iloc[j].values)
            cluster_count = len(cluster_cids_i)
            print("num: " + str(cluster_count))
            #print(cluster_cids_i)
            cid_occurrences = list()
            # get counts of each ON WG IDENTIFIER
            for cid in unique_cids:
                cid_occurrences.append(cluster_cids_i.count(cid))
            #print(cid_occurrences)
            
            score = float(max(cid_occurrences))/float(cluster_count)
            print(score)
            scores.append(score)


    def get_sil_score(self):
        return metrics.silhouette_score(self.train_x, self.labels)

    def get_db_idx_score(self):
        return davies_bouldin_score(self.train_x, self.labels)

    def get_labels(self):
        return self.labels

class Birch_(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.birch_model = Birch(n_clusters=num_clusters).fit(train_x)
        self.birch_model.predict(train_x)
        self.labels = self.birch_model.labels_
        self.num_clusters = num_clusters

class GMM(ClusterModel):
    def __init__(self, n_components, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(train_x)
        self.labels = self.gmm_model.predict(train_x)

class Hierarchical(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.hierarchical_model = AgglomerativeClustering(n_clusters=num_clusters).fit(train_x)
        self.labels = self.hierarchical_model.labels_
        self.num_clusters = num_clusters

    def _get_cluster_id(self, queries):
        pred_y = list()
        weights = list()
        temp_x = self.train_x
        print(temp_x.shape)
        # get the cluster id for a given set of queries
        for query in queries:
            tokens = self._tokenize(query)
            if self.rep == "bow":
                w, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
            elif self.rep == "tfidf":
                w, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
                print("tfidf")
            temp_x = np.append(temp_x, w.toarray(), axis=0)
            feature_idx = np.where(w.toarray()[0] > 0)[0]
            print(feature_idx)
            entry = list()
            for x in feature_idx:
                print(x)
                entry.append(self.feature_names[x])
            print(entry)
            weights.append(w.toarray())
        # fit model on entire dataset including queries
        print(temp_x.shape)
        print(temp_x[:10])
        hierarchical_model_1 = AgglomerativeClustering(n_clusters=self.num_clusters).fit(temp_x)
        # get predictions
        pred_y = hierarchical_model_1.labels_[-len(queries):]
        print(pred_y)
        return pred_y, weights

    def get_nearest_neighbours(self, queries):
        # get cluster id
        pred_ys, weights = self._get_cluster_id(queries)

        # tokenize data and get bag of words
        #tokens = self._tokenize(query)
        #weights, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
        #print(weights)

        for i in range(len(queries)):
            pred_y = pred_ys[i]
            weight = weights[i]
            query = queries[i]
            print("pred_y = " + str(pred_y) + ", weight = " + str(weight) + ", query = " + str(query))

            # get all training values in same cluster
            train_x_idx = np.where(self.labels == pred_y)
            cluster_set = self.train_x[np.where(self.labels == pred_y)]
            cluster_set_y = self.train_y.iloc[train_x_idx[0]].as_matrix()
            print(queries[i] + " clustered with: " + str(cluster_set.shape[0]))

            # get distances between query and all in same cluster
            distances = [euclidean_distances(weight, train_example.reshape(1, -1)) for train_example in cluster_set]

            # sort by distance
            sorted_dist = [d for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
            sorted_by_dist = [x for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
            sorted_y_by_dist = [x for d,x in sorted(zip(distances,cluster_set_y), key=lambda x: x[0])]

            # print top 20% closest neighbours
            num_in_cluster = float(cluster_set.shape[0])
            top_20_percent = int(math.ceil(num_in_cluster*0.20))
            if (num_in_cluster < 10):
                top_20_percent = num_in_cluster
            neighbours = list()
            for i in range(int(top_20_percent)):
                #print(sorted_by_dist[i])
                entry = list()

                feature_idx = np.where(cluster_set[i] > 0)[0]
                for x in feature_idx:
                    entry.append(self.feature_names[x])
                neighbours.append(entry)
                #print(entry) 
                print(cluster_set_y[i])


class Kmeans(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.kmeans_model = KMeans(n_clusters=num_clusters, random_state=0).fit(train_x)
        self.labels = self.kmeans_model.labels_
        self.num_clusters = num_clusters

    def get_nearest_neighbours(self, query):
        print("Search query:" + query)
        # get cluster id
        pred_y, weights = self._get_cluster_id(query)
        print(pred_y)

        # get all training values in same cluster
        train_x_idx = np.where(self.labels == pred_y)
        cluster_set = self.train_x[np.where(self.labels == pred_y)]
        cluster_set_y = self.train_y.iloc[train_x_idx[0]].as_matrix()
        print("Clustered with: " + str(cluster_set.shape[0]))

        # get distances between query and all in same cluster
        distances = [euclidean_distances(weights, train_example.reshape(1, -1)) for train_example in cluster_set]

        # sort by distance
        sorted_dist = [d for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
        sorted_by_dist = [x for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
        sorted_y_by_dist = [x for d,x in sorted(zip(distances,cluster_set_y), key=lambda x: x[0])]

        # print top 20% closest neighbours
        num_in_cluster = float(cluster_set.shape[0])
        top_20_percent = int(math.ceil(num_in_cluster*0.20))
        if (num_in_cluster < 10):
            top_20_percent = num_in_cluster
        neighbours = list()
        for i in range(int(num_in_cluster)):
            #print(sorted_by_dist[i])
            entry = list()
            feature_idx = np.where(cluster_set[i] > 0)[0]
            for x in feature_idx:
                entry.append(self.feature_names[x])
            if not query in str(entry):
                neighbours.append(entry)
                print(cluster_set_y[i])

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = self._tokenize(query)
        print(type(self.train_x))

        # get representation (bag of words)
        pred_y =""
        if self.rep == "bow":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        elif self.rep == "tfidf":
            weights, y, _ = tokens_to_bagofwords([tokens, ], 1, TfidfVectorizer, self.feature_names)
        pred_y = self.kmeans_model.predict(weights)
        return pred_y, weights

class Meanshift(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.meanshift_model = MeanShift().fit(train_x)
        self.labels = self.meanshift_model.labels_
        self.num_clusters = num_clusters

class Spectral(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.spectral_model = SpectralClustering(n_clusters=num_clusters,
                                                 random_state=0,
                                                 assign_labels='discretize').fit(train_x)
        self.labels = self.spectral_model.labels_
        self.num_clusters = num_clusters

class Affinity(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.affinity_model = AffinityPropagation().fit(train_x)
        self.labels = self.affinity_model.labels_
        self.num_clusters = num_clusters

