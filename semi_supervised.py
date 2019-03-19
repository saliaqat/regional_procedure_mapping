from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from tqdm import tqdm
import keras.backend as K
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
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
from sklearn import preprocessing
from data_reader import DataReader
from data_manipulator import *
import warnings
warnings.filterwarnings("ignore")

# shuffle in unison
def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    np.random.seed(42)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


# get data
data_reader = DataReader()
df = data_reader.get_all_data()
train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=True)
train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)

tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, feature_names=feature_names)

# encode the labels into smaller integers rather than large integers
le = preprocessing.LabelEncoder()
le.fit(np.concatenate((test_y.values, train_y.values)))
train_y = le.transform(train_y.values)
test_y = le.transform(test_y.values)

# combine train and test
x = np.concatenate((train_x.todense(), test_x.todense()))
y = np.concatenate((train_y, test_y))

# reduce the dimension using autoencoder
#encoder = load_model('encoder-1000.h5')
#x = encoder.predict(x)

# this function creates pair between same class and different class with
# appropriate targets    
def create_pair(labels_dict):
    
    dataset_sim = []
    targets_sim = []
    for label in sorted(labels_dict):
        samples = labels_dict[label]
        for i in range(len(samples)):
            for j in range(len(samples)):
                dataset_sim.append([samples[i], samples[j]])
                targets_sim.append(1)

    dataset_diff = []
    targets_diff = []
    for labelA in sorted(labels_dict):
        for labelB in sorted(labels_dict):
            if labelA != labelB:
                samplesA = labels_dict[labelA]
                samplesB = labels_dict[labelB]
                for i in range(len(samplesA)):
                    for j in range(len(samplesB)):
                        dataset_diff.append([samplesA[i], samplesB[j]])
                        targets_diff.append(0)
                        
    idx = np.random.permutation(len(dataset_sim))    
    return dataset_sim + list(np.array(dataset_diff)[idx]), \
                    targets_sim + list(np.array(targets_diff)[idx])
                                         
    
# create dataset for siamese neural network
def create_pairwise_dataset(x, y, k=8, train_ratio=0.6, limit=50):
    # k is the minimum sample needed for each class
    # train ratio says split the data into 80:20
        
    labels_dict = {}
    for i in range(len(y)):
        if y[i] in labels_dict:
            # cap at only k samples per class
            if len(labels_dict[y[i]]) <= k:
                labels_dict[y[i]].append(x[i])
        else:
            labels_dict[y[i]] = [x[i]]

    labels_dict_train = {}
    labels_dict_test = {}
    counter = 1
    for label in sorted(labels_dict):
        if counter > limit:
            break
        if len(labels_dict[label]) >= k:
            labels_dict_train[label] = labels_dict[label][:int(k*train_ratio)]
            labels_dict_test[label] = labels_dict[label][int(k*train_ratio):k]
            
        counter += 1
         
    # delete the other samples 
    del labels_dict
    
    print('Creating dataset with total classess of {} ..'.format(len(labels_dict_train)))
        
    train_x, train_y = create_pair(labels_dict_train)
    
    print('Total samples created for training: {}'.format(len(train_y)))
    
    test_x, test_y = create_pair(labels_dict_test)
    
    print('Total samples created for testing: {}'.format(len(test_y)))
    
    return train_x, test_x, train_y, test_y

train_x, test_x, train_y, test_y = 0, 0, 0, 0

train_paired, test_paired, train_paired_target, test_paired_target = create_pairwise_dataset(x, y)

# set the data as separate numpy array
pair1_train = []
pair2_train = []
for i in range(len(train_paired)):
    pair1_train.append(np.array(train_paired[i][0]))
    pair2_train.append(np.array(train_paired[i][1]))
    
pair1_train = np.array(pair1_train)
pair2_train = np.array(pair2_train)
train_paired_target = np.array(train_paired_target)

# shuffle in unison
pair1_train, pair2_train, train_paired_target = unison_shuffled_copies(pair1_train, pair2_train, train_paired_target)
  
pair1_test = []
pair2_test = []
for i in range(len(test_paired)):
    pair1_test.append(np.array(test_paired[i][0]))
    pair2_test.append(np.array(test_paired[i][1]))
    
pair1_test = np.array(pair1_test)
pair2_test = np.array(pair2_test)
test_paired_target = np.array(test_paired_target)

# shuffle in unison
pair1_test, pair2_test, test_paired_target = unison_shuffled_copies(pair1_test, pair2_test, test_paired_target)
 
pair1_train = pair1_train.reshape(-1, 24472)
pair2_train = pair2_train.reshape(-1, 24472)  
pair1_test = pair1_train.reshape(-1, 24472)
pair2_test = pair2_train.reshape(-1, 24472)   

#np.save(pair1_train)
#np.save(pair2_train)
#np.save(pair1_test)
#np.save(pair2_test)
#np.save(train_paired_target)
#np.save(test_paired_target)

# siamese neural network structure 
input_shape = pair1_train.shape[1]
left_input = Input(shape=(input_shape,))
right_input = Input(shape=(input_shape,))

nnet = Sequential()
nnet.add(Dense(500, activation='relu', input_dim=input_shape))
nnet.add(Dense(250, activation='relu'))
nnet.add(Dense(100, activation='relu'))
nnet.add(Dense(50, activation="sigmoid"))

#encode each of the two inputs into a vector with the convnet
encoded_l = nnet(left_input)
encoded_r = nnet(right_input)

#merge two encoded inputs with the l1 distance between them
l1_distance = lambda x: K.abs(x[0]-x[1])
merge = Lambda(l1_distance)([encoded_l, encoded_r])
merge1 = Dense(1,activation='sigmoid')(merge)
siamese_net = Model(input=[left_input,right_input],output=merge1)
siamese_net.compile(loss="binary_crossentropy",optimizer='adam', metrics=['accuracy'])
print(siamese_net.summary())

siamese_net.fit([pair1_train, pair2_train], train_paired_target, 
                validation_data=([pair1_test, pair2_test], test_paired_target),
                epochs=100, batch_size=500, shuffle=True)

siamese_net.save('siamese.h5')

# evaluation for siamese network few-shot learning technique
#for i in range(len(pair1_test)):
#    outputs = []
#    for j in range(len(pair2_test))
#    pair1_test[i]


