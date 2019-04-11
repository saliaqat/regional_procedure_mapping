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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))

class Kmeans(ClusterModel):
    def __init__(self, num_clusters, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.kmeans_model = KMeans(n_clusters=num_clusters, random_state=0).fit(train_x)
        self.labels = self.kmeans_model.labels_
        self.num_clusters = num_clusters

    def get_nearest_neighbours(self, query):
        print("Search query:" + query)
        # get cluster id
        pred_y = self._get_cluster_id(query)

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
        for i in range(int(top_20_percent)):
            #print(sorted_by_dist[i])
            entry = list()
            for x in np.where(cluster_set[i].toarray() != 0)[1]:
                entry.append(self.feature_names[x])
            neighbours.append(entry)
            #print(entry) 
            print(cluster_set_y[i])

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

