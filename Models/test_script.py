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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from Models.birch import Birch_
from Models.hierarchial import Hierarchial

def main():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    y = df['ON WG IDENTIFIER'].values
    df.drop(['src_file', 'ON WG IDENTIFIER'], 1, inplace=True)
    tokens, y = tokenize_columns(df, y, save_missing_feature_as_string=False, remove_repeats=True,
                                remove_num=True)
    x, y, feature_names = tokens_to_bagofwords(tokens, y)

    #tfid
    corpus = list(map(' '.join, tokens[:]))
    vectorizer = TfidfVectorizer()
    mat = vectorizer.fit_transform(corpus)

    #    def __init__(self, num_clusters, feature_names, train_x, train_y):
    b = Birch_(10, feature_names, mat, y)
    print('d b score: ' + str(b.get_db_idx_score()))
    print('sil score: ' + str(b.get_sil_score()))

    h = Hierarchial(10, feature_names, mat, y)
    print('d b score: ' + str(h.get_get_db_idx_score()))
    print('sil score: ' + str(h.get_sil_score()))



if __name__ == '__main__':
    main()
