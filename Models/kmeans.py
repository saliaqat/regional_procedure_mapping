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
import math
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))

class Kmeans(Model):
    def __init__(self, num_clusters, feature_names, train_x, train_y):
        self.kmeans_model = KMeans(n_clusters=num_clusters, random_state=0).fit(train_x)
        self.feature_names = feature_names
        self.train_x = train_x
        self.train_y = train_y
        self.labels = self.kmeans_model.labels_
        self.num_clusters = num_clusters
        self.sil_score = -100.0
        self.db_idx_score = -100.0

    def _tokenize(self, query):
        regex_string=r'[a-zA-Z]+'
        tokenizer = RegexpTokenizer(regex_string)
        tokens = tokenizer.tokenize(query.lower())
        tokens = [ x for x in tokens if x.isalpha()]
        tokens = [ x for x in tokens if len(x) > 2 ]
        return tokens

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = self._tokenize(query)
        
        # get representation (bag of words)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        pred_y = self.kmeans_model.predict(weights)
        
        return pred_y

    def get_nearest_neighbours(self, query):
        print("Search query:" + query)
        # get cluster id
        pred_y = self._get_cluster_id(query)

        # tokenize data and get bag of words
        tokens = self._tokenize(query)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        #print(weights)

        # get all training values in same cluster
        train_x_idx = np.where(self.labels == pred_y)
        cluster_set = self.train_x[np.where(self.labels == pred_y)]
        print("Clustered with: " + str(cluster_set.shape[0]))

        # get distances between query and all in same cluster
        distances = [euclidean_distances(weights, train_example.reshape(1, -1)) for train_example in cluster_set]

        # sort by distance
        sorted_dist = [d for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
        sorted_by_dist = [x for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]

        # print top 20% closest neighbours
        num_in_cluster = float(cluster_set.shape[0])
        top_20_percent = int(math.ceil(num_in_cluster*0.20))
        if (num_in_cluster < 10):
            top_20_percent = num_in_cluster
        neighbours = list()
        for i in range(int(top_20_percent)):
            #print(sorted_by_dist[i])
            entry = list()
            for x in np.where(cluster_set[i].toarray() != 0)[1]:
                entry.append(self.feature_names[x])
            neighbours.append(entry)
            print(entry) 

    def eval(self):
        self.sil_score = metrics.silhouette_score(self.train_x, self.labels, metric='euclidean')
        self.db_idx_score = metrics.davies_bouldin_score(self.train_x, self.labels)
        #print(self.sil_score)
        #print(self.db_idx_score)

        # evaluate with ON WG IDENTIFIER
        self.custom_score()

    def custom_score(self):
        # evaluate with ON WG IDENTIFIER
        # how to get the number of ids
        # for each cluster count the number onwgid occurences
        # get unique list of cluster ids
        unique_cids = self.train_y['ON WG IDENTIFIER'].unique()
        print(len(unique_cids))

        # for each cluster
        '''scores = list() 
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
            print("-------------------")
        self.plot_custom_score(scores)'''

    def plot_custom_score(self, scores):
        # the histogram of the data
        plt.hist(scores, 50, density=True, facecolor='g', alpha=0.75)

        plt.xlabel('scores')
        plt.ylabel('frequency')
        plt.title('Histogram of Custom Scores')
        plt.grid(True)
        plt.savefig('custom_scores.png')

    def get_sil_score(self):
        if self.sil_score < -10:
            self.eval()
        return self.sil_score

    def get_db_idx_score(self):
        if self.db_idx_score < -10:
            self.eval()
        return self.db_idx_score

    def get_labels(self):
        return self.labels
