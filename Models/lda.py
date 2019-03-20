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
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))

class Lda(Model):
    def __init__(self, train_x_raw, train_y_raw, NUM_TOPICS, passes=15):
        self.corpus, self.dictionary, self.text_data = self._prep_lda_data(train_x_raw, train_y_raw)
        self.NUM_TOPICS = NUM_TOPICS
        self.passes = passes

    def tokenize(self, train_x_raw, train_y_raw):
        tokens, _ = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_num=True, remove_short=True, remove_empty=True)
        return tokens

    def _prep_lda_data(self, train_x_raw, train_y_raw):
        # convert to bag of words
        data = self.tokenize(train_x_raw, train_y_raw)
        dictionary = corpora.Dictionary(data)
        corpus = [dictionary.doc2bow(text) for text in data]
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dict.gensim')
        return corpus, dictionary, data

    def train(self):
        ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics = self.NUM_TOPICS, id2word=self.dictionary, passes=self.passes)
        ldamodel.save('model' + str(self.NUM_TOPICS) +'.gensim')
        print("finished running lda!")
        print(ldamodel.show_topics(num_topics=self.NUM_TOPICS, num_words=30, log=False, formatted=True))
        self.compute_coherence(ldamodel, self.text_data, self.dictionary)
   
    def compute_coherence(self, lda_model,text_data,dictionary):
        # Compute Coherence Score using c_v
        coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        return coherence_lda

