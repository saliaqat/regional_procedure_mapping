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
import pickle
import seaborn as sns
from data_manipulator import *
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings

warnings.filterwarnings("ignore")
# en_stop = set(nltk.corpus.stopwords.words('english'))
from Models.model import Model
from Models.clustermodel import ClusterModel
from sklearn import mixture
from sklearn.metrics import davies_bouldin_score

class GMM(ClusterModel):
    def __init__(self, n_components, feature_names, train_x, train_y, rep):
        ClusterModel.__init__(self, train_x, train_y, feature_names, rep)
        self.gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(train_x)
        self.labels = self.gmm_model.predict(train_x)
