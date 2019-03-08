#from __future__ import print_function, division
#from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda
#from keras.layers import BatchNormalization, Activation
#from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.convolutional import UpSampling2D, Conv1D, MaxPooling1D
#from keras.models import Sequential, Model, load_model
#from keras.optimizers import RMSprop, Adam
#import keras.applications as apps
#from functools import partial
#import requests
#from io import BytesIO
#from tqdm import tqdm
#import keras.backend as K
#import pickle
#import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
import pandas as pd
from Models.logistic_regression import BinaryLogisticRegressionModel, MultiClassLogisticRegression
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_reader import DataReader
from data_manipulator import *
import warnings
warnings.filterwarnings("ignore")

## get all the labelled data
#reader = DataReader()
#data = reader.get_all_data()
#
#print('\n================== Before removing missing values ==================\n')
#print('No. of samples: {}'.format(len(data)))
#print('No. of classes: {}'.format(data['ON WG IDENTIFIER'].nunique()))
#counts = data.groupby(['ON WG IDENTIFIER']).size().to_frame(name='counts') \
#                        .sort_values(['counts']).values
#print('Max no. of samples for a class: {}'.format(counts[-1][-1]))
#print('Min no. of samples for a class: {}'.format(counts[0][0]))
#print('Avg no. of samples for a class: {}'.format(round(np.mean(counts), 2)))
#print('\n===================================================================\n')
#
#print('================== After removing missing values ====================\n')
## drop rows with missing values
#dataNoNan = data.dropna()
#print('No. of samples: {}'.format(len(dataNoNan)))
#print('No. of classes: {}'
#      .format(dataNoNan['ON WG IDENTIFIER'].nunique()))
#
#counts = dataNoNan.groupby(['ON WG IDENTIFIER']).size().to_frame(name='counts') \
#                        .sort_values(['counts']).values
#print('Max no. of samples for a class: {}'.format(counts[-1][-1]))
#print('Min no. of samples for a class: {}'.format(counts[0][0]))
#print('Avg no. of samples for a class: {}'.format(round(np.mean(counts), 2)))
#print('\n====================================================================')
#
#
#data = data.drop(['src_file'], axis=1)
#targets = data['ON WG IDENTIFIER'].values
#old_features = data.drop(['ON WG IDENTIFIER'], axis=1).values
#features = []
#for i in range(len(old_features)):
#    features.append(' '.join(str(i) for i in old_features[i]))

print('Fitting logistic regression with original features ...')
data_reader = DataReader()
df = data_reader.get_all_data()
train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=True)
train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)
tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, feature_names=feature_names)
print('Train shape: ', train_x.shape)
print('Test shape: ', test_x.shape)
#lg = MultiClassLogisticRegression()
#lg.train(train_x, train_y)
#print('Accuracy with original features: {}'.format(lg.score(test_x, test_y)))

print('Reducing the dimension of the features ...')
# reduce sparse matrix to a compact one through SVD
svd = TruncatedSVD(n_components=5000, random_state=42)
train_comp_x = svd.fit_transform(train_x) 
test_comp_x = svd.transform(test_x) 
print('Train compressed shape: ', train_comp_x.shape)
print('Test compressed shape: ', test_comp_x.shape)
print('Fitting logistic regression with compressed features: {} ...'.format(5000))
lg = MultiClassLogisticRegression()
lg.train(train_comp_x, train_y)
print('Accuracy with compressed features of dim {}: {}'.format(5000, lg.score(test_comp_x, test_y)))


#data_reader = DataReader()
#df = data_reader.get_all_data()
#train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
#tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=True)
#train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)
#tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
#test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)
#print('Train shape: ', train_x.shape)
#print('Test shape: ', test_x.shape)
#
## reduce the dimension of the tfidf matrix
#print('Applying truncatedSVD to sparse matrix ...')
#svd = TruncatedSVD(n_components=5000, random_state=42)
#train_x = svd.fit_transform(train_x) 
#test_x = svd.transform(test_x) 
#
##train_x = np.random.rand(44120, 5000)
##test_x = np.random.rand(14707, 5000)
#
#print('Train shape: ', train_x.shape)
#print('Test shape: ', test_x.shape)
#
#def create_pairwise_dataset(x, y, samples_from_each_class, mode='training'):
#        
#    # create dataset for siamese network
#    y = y.values
#    labels_dict = {}
#    for i in range(len(y)):
#        if y[i][0] in labels_dict:
#            # cap at only 10 samples per class
#            if len(labels_dict[y[i][0]]) <= samples_from_each_class:
#                labels_dict[y[i][0]].append(x[i])
#        else:
#            labels_dict[y[i][0]] = [x[i]]
#    
#    dataset = []
#    targets = []
#    new_labels_dict = {}
#    for label in labels_dict:
#        if 10 <= len(labels_dict[label]) <= 15:
#            new_labels_dict[label] = labels_dict[label]
#    labels_dict = new_labels_dict
#    print('Creating dataset for {} with total classess of {} ..'.format(mode, len(labels_dict)))
#    for label in labels_dict:
#        samples = labels_dict[label]
#        for i in range(len(samples)):
#            for j in range(len(samples)):
#                dataset.append([samples[i], samples[j]])
#                targets.append(1)
#                
#    for labelA in labels_dict:
#        for labelB in labels_dict:
#            if labelA != labelB:
#                samplesA = labels_dict[labelA]
#                samplesB = labels_dict[labelB]
#                for i in range(len(samplesA)):
#                    for j in range(len(samplesB)):
#                        dataset.append([samplesA[i], samplesB[j]])
#                        targets.append(0)
#                        
#    print('Total samples created for {}: {}'.format(mode, len(targets)))
#    
#    return dataset, targets
#
#train_paired, train_target = create_pairwise_dataset(train_x, train_y, 10, 'training')
#test_paired, test_target = create_pairwise_dataset(test_x, test_y, 10, 'testing')
#
## create a siamese neural network   
#input_shape = (train_x.shape[1],)
#left_input = Input(input_shape)
#right_input = Input(input_shape)
#
#nnet = Sequential()
#nnet.add(Dense(2500, activation='relu', input_dim=input_shape[0]))
#nnet.add(Dense(1000, activation='relu'))
#nnet.add(Dense(500, activation='relu'))
#nnet.add(Dense(100, activation="sigmoid"))
#
##encode each of the two inputs into a vector with the convnet
#encoded_l = nnet(left_input)
#encoded_r = nnet(right_input)
#
##merge two encoded inputs with the l1 distance between them
#l1_distance = lambda x: K.abs(x[0]-x[1])
#merge = Lambda(l1_distance)([encoded_l, encoded_r])
#merge1 = Dense(1,activation='sigmoid')(merge)
#siamese_net = Model(input=[left_input,right_input],output=merge1)
#siamese_net.compile(loss="binary_crossentropy",optimizer='adam')
#print(siamese_net.summary())
#
#epochs = 10
#batch_size = 5000
#num_batches = int(len(train_paired) / batch_size)
#pbar = tqdm(total=epochs * num_batches)
#print('Num batches: ',num_batches)
#for e in range(1, epochs+1):
#    
#    for i in range(num_batches):
#        
#        pbar.update(1)
#        
#        # Select a random batch of images
#        inputs = train_paired[i * batch_size: (i + 1) * batch_size]
#        targets = train_target[i * batch_size: (i + 1) * batch_size]
#        inputs1 = []
#        inputs2 = []
#        for j in range(batch_size):
#            inputs1.append(inputs[j][0])
#            inputs2.append(inputs[j][1])
#            
#        inputs1 = np.array(inputs1).reshape(batch_size, input_shape[0])
#        inputs2 = np.array(inputs2).reshape(batch_size, input_shape[0])
#        loss = siamese_net.train_on_batch(x=[inputs1, inputs2], y=targets)
#
#        if i % 100 == 0:
#            print('{} Loss: {}'.format(e, loss))
#            
#    if e >= 5:
#        siamese_net.save('model-' + str(e) + '.h5')
#        print('Model saved for current epoch')
#
#
