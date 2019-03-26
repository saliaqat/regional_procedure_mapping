import matplotlib as MPL
MPL.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import pairwise_distances
from sklearn import metrics
import pandas as pd
from data_reader import DataReader
from data_manipulator import *
from sklearn.model_selection import train_test_split
import keras
import gzip
import nltk
#nltk.download('stopwords')
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from Models.lda import Lda
from Models.kmeans import Kmeans
from Models.dbscan import DBscan
from Models.birch import Birch_
from Models.hierarchial import Hierarchial


import argparse

def plot_dataset_size_per_site(dataset_sizes, x_labels):
    fig = plt.figure()
    y_pos = np.arange(len(x_labels))
    plt.bar(y_pos, dataset_sizes)
    plt.xticks(y_pos, x_labels, rotation='vertical')
    plt.tight_layout()
    plt.title('Data set sizes by Site')
    plt.savefig("dataset_sizes_per_site.pdf", bbox_inches = "tight")

def plot_sil_scores_per_site(scores, x_labels):
    fig = plt.figure()
    y_pos = np.arange(len(x_labels))
    plt.bar(y_pos, scores)
    plt.xticks(y_pos, x_labels, rotation='vertical')
    plt.tight_layout()
    plt.title('Silhouette Scores for Clustering by Site')
    plt.savefig("sil_scores_per_site.pdf", bbox_inches = "tight")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run unsupervised methods', add_help=False)
    parser.add_argument("-h", "--help",  action="store_true", dest="help")
    parser.add_argument("-m", "--model", action="store", required=True, dest="MODEL", choices=['kmeans', 'lda', 'dbscan', 'birch', 'hierarchical'], help="Run model ")
    parser.add_argument("-r", "--rep",   action="store", required=True, dest="REP", choices=['bow', 'tfidf'], help="Use bag of words representation (BOW), or tfidf representation")
    args = parser.parse_args()

	# get data
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
    train_x_raw = pd.concat([train_x_raw, test_x_raw], axis=0)
    train_y_raw = pd.concat([train_y_raw, test_y_raw], axis=0)
    train_x_raw.drop(['RIS PROCEDURE CODE'], axis=1, inplace=True)
    
    # tokenize and subsample
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z0-9]+', 
        save_missing_feature_as_string=False, remove_short=True, remove_num=True, remove_empty=True)
    print("done tokenizing columns")

    # get representation of data
    feature_names = list()
    train_x = list()
    train_y = list()
    if args.REP == "bow":
        train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, CountVectorizer)
        print("done converting to bag of words representation")
    elif args.REP == "tfidf":
        train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, TfidfVectorizer)
        print("done converting to tfidf representation")

    # run models
    print(str(args.MODEL) + " results: ")
    if args.MODEL == "kmeans":
        num_clusters = 1500
        kmeans = Kmeans(num_clusters, feature_names, train_x, train_y)
        kmeans.eval()
        print("sil score:" + str(kmeans.get_sil_score()))
        print("db idx:" + str(kmeans.get_db_idx_score()))
        print("getting nearest: ")
        kmeans.get_nearest_neighbours("Y DIR - ANGIOGRAM")
        kmeans.get_nearest_neighbours("US KNEE BIOPSY/ASPIRATION")
        kmeans.get_nearest_neighbours("G TUBE INSERTION")
    elif args.MODEL == "lda":
        # run lda
        lda = Lda(train_x_raw, train_y_raw, 1500, passes=15)
        lda.train()
        print("finished running lda")
    elif args.MODEL == "dbscan":
        # run dbscan
        dbscan = DBscan(10, feature_names, train_x, train_y)
        dbscan.eval()
        print("sil score:" + str(dbscan.get_sil_score()))
        print("db idx:" + str(dbscan.get_db_idx_score()))
    elif args.MODEL == "birch":
        b = Birch_(10, feature_names, train_x, train_y)
        print('sil score: ' + str(b.get_sil_score())) 
        print('db idx: ' + str(b.get_db_idx_score()))
    elif args.MODEL == "hierarchical":
        h = Hierarchial(10, feature_names, train_x, train_y)
        print('sil score: ' + str(h.get_sil_score())) 
        print('db idx: ' + str(h.get_db_idx_score()))



if __name__ == '__main__':
    main()
