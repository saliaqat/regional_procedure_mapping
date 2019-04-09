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
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.pairwise import euclidean_distances
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