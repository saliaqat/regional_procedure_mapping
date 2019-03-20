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

def run_models(train_x, train_y, feature_names):
    # run clustering
    num_clusters = 1500
    cluster_full_set(num_clusters, train_x, train_y, feature_names)

def cluster_full_set(num_clusters,train_x, train_y, feature_names):
    nc = num_clusters

    kmeans = Kmeans(n_clusters=nc, random_state=0).fit(train_x)
    labels = kmeans.labels_

    cluster = Cluster(kmeans, feature_names, train_x, labels)
    
    #print("getting nearest")
    #cluster.get_nearest_neighbours("Y DIR - ANGIOGRAM")

    sil_score = metrics.silhouette_score(train_x, labels, metric='euclidean')
    db_idx_score = metrics.davies_bouldin_score(train_x, labels)
    #scores.append(sil_score)
    #label = s.lstrip("input_data/").rstrip(".csv")
    #    x_labels.append(label)
    #    dataset_sizes.append(train_x.shape[0])
    print(sil_score)
    print(db_idx_score)
    #plot_sil_scores_per_site(scores, x_labels)
    #plot_dataset_size_per_site(dataset_sizes, x_labels)


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
	# get data
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
    train_y_raw = train_x_raw['RIS PROCEDURE CODE']
    test_y_raw = test_x_raw['RIS PROCEDURE CODE']
    train_x_raw = pd.concat([train_x_raw, test_x_raw], axis=0)
    train_y_raw = pd.concat([train_y_raw, test_y_raw], axis=0)
    train_x_raw = train_x_raw.drop("RIS PROCEDURE CODE", axis=1)
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z]+', 
        save_missing_feature_as_string=False, remove_short=True, remove_num=False, remove_empty=True)

    # run with regular bag of words
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, CountVectorizer)
    num_clusters = 1500
    kmeans = Kmeans(num_clusters, feature_names, train_x)
    labels = kmeans.labels_
    sil_score = metrics.silhouette_score(train_x, labels, metric='euclidean')
    db_idx_score = metrics.davies_bouldin_score(train_x, labels)
    print(sil_score)
    print(db_idx_score)
    print("finished running knn with countvectorizer")

    print("getting nearest")
    kmeans.get_nearest_neighbours("Y DIR - ANGIOGRAM")

    # run with tfidf
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, TfidfVectorizer)
    num_clusters = 1500
    kmeans = Kmeans(num_clusters, feature_names, train_x)
    labels = kmeans.labels_
    sil_score = metrics.silhouette_score(train_x, labels, metric='euclidean')
    db_idx_score = metrics.davies_bouldin_score(train_x, labels)
    print(sil_score)
    print(db_idx_score)
    print("finished running knn with tfidfvectorizer")

    # run lda
    lda = Lda(train_x_raw, train_y_raw, 1500, passes=15)
    lda.train()
    print("finished running lda")

if __name__ == '__main__':
    main()
