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
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))

class Kmeans(Model):
    def __init__(self, num_clusters, feature_names, train_x):
        self.kmeans_model = KMeans(n_clusters=num_clusters, random_state=0).fit(train_x)
        self.feature_names = feature_names
        self.train_x = train_x
        self.labels = self.kmeans_model.labels_

    def _tokenize(self, query):
        regex_string=r'[a-zA-Z]+'
        tokenizer = RegexpTokenizer(regex_string)
        tokens = tokenizer.tokenize(query.lower())
        return tokens

    def _get_cluster_id(self, query):
        # get the cluster id for a given query
        # get prediction for a given query
        tokens = _tokenize(query)
        
        # get representation (bag of words)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)
        pred_y = self.kmeans_model.predict(weights)
        
        return pred_y

    def get_nearest_neighbours(self, query):
        # get cluster id
        pred_y = self._get_cluster_id(query)

        # tokenize data and get bag of words
        tokens = _tokenize(query)
        weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, self.feature_names)

        # get all training values in same cluster
        train_x_idx = np.where(self.train_y == pred_y)
        cluster_set = self.train_x[np.where(self.train_y == pred_y)]
        print("Clustered with: " + str(cluster_set.shape[0]))

        # get distances between query and all in same cluster
        distances = [euclidean_distances(weights, train_example.reshape(1, -1)) for train_example in cluster_set]

        # sort by distance
        sorted_dist = [d for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
        sorted_by_dist = [x for d,x in sorted(zip(distances,cluster_set), key=lambda x: x[0])]
        sorted_idx_by_dist = [x for d,x in sorted(zip(distances,train_x_idx))]

        # print the closest neighbours
        for i in range(3):
            idx = sorted_idx_by_dist[0][i]
            print(idx)
            print(cluster_set[idx])


    def get_labels(self):
        return self.labels
