from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1
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

tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_empty=True, remove_num=True, remove_repeats=True, remove_short=True)
train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, vectorizer_class=CountVectorizer)

tokens, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, save_missing_feature_as_string=False, remove_empty=True, remove_num=True, remove_repeats=True, remove_short=True)
test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, vectorizer_class=CountVectorizer, feature_names=feature_names)

# encode the labels into smaller integers rather than large integers
le = preprocessing.LabelEncoder()
le.fit(np.concatenate((test_y.values, train_y.values)))
train_y = le.transform(train_y.values)
test_y = le.transform(test_y.values)

# combine train and test
x = np.concatenate((train_x.todense(), test_x.todense()))
#x = np.load('transformed_x.npy')
y = np.concatenate((train_y, test_y))

print('Max: ', np.max(x))
print('Min: ', np.min(x))

# no. of classes
import sys
variable1 = int(sys.argv[1])
limit = variable1
variable2 = int(sys.argv[2])
topk_var = variable2
variable3 = int(sys.argv[3])
zero_shot = variable3
print('Limit: ', limit)
print('Top: ', topk_var)
print('Zero shot classes: ', zero_shot)

    
# this function creates pair between same class and different class with
# appropriate targets    
def create_pair(labels_dict, labels, train):
    
    dataset_sim = []
    targets_sim = []
    for label in sorted(labels_dict):
        samples = labels_dict[label]
        for i in range(len(samples)):
            for j in range(len(samples)):
                dataset_sim.append([samples[i], samples[j]])
                targets_sim.append(0)
    
    dataset_diff = []
    targets_diff = []
    
    if train:
        ratio = 1
    else:
        ratio = 1
        
    import random
    while len(targets_diff) != int(len(targets_sim)*ratio):
        idx1 = random.choice(list(labels_dict.keys()))
        idx2 = random.choice(list(labels_dict.keys()))
        if idx1 != idx2:
            samplesA = labels_dict[idx1]
            samplesB = labels_dict[idx2]
            a = np.random.randint(len(samplesA))
            b = np.random.randint(len(samplesB))
            dataset_diff.append([samplesA[a], samplesB[b]])
            targets_diff.append(1)
            
            
    return dataset_sim + dataset_diff, targets_sim + targets_diff

# create dataset for siamese neural network
def create_pairwise_dataset(x, y, k=15, train_ratio=0.66, limit=limit):
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
    labels = []
    zero_shot_test_samples = []
    zero_shot_test_labels = []
    zero_shot_set = {}
    counter = 0
    zero_idx = np.arange(500, 500+zero_shot)#np.random.permutation(np.arange(500, 900))[:zero_shot]
    m = 0
    for label in sorted(labels_dict):
        if counter > limit + zero_shot:
            break
        elif counter > limit:
            if len(labels_dict[zero_idx[m]]) >= k:
                zero_shot_set[zero_idx[m]] = labels_dict[zero_idx[m]][:int(np.ceil(k*train_ratio))]
                samples = labels_dict[zero_idx[m]][int(np.ceil(k*train_ratio)):k]
                for s in samples:
                    zero_shot_test_samples.append(s)
                    zero_shot_test_labels.append(zero_idx[m])
                m += 1
        else:
            if len(labels_dict[label]) >= k: 
                labels_dict_train[label] = labels_dict[label][:int(np.ceil(k*train_ratio))]
#                labels_dict_test[label] = labels_dict[label][int(np.ceil(k*train_ratio)):k]
                labels.append(label)
        counter += 1
        
    counter = 0
    test_limit = int(limit)
    for label in sorted(labels_dict):
        if counter > test_limit:
            break
        if (label in labels_dict_train) or (label in zero_shot_set):
            pass
        else:
            labels_dict_test[label] = labels_dict[label][:int(np.ceil(k*train_ratio))]
            counter += 1
            
    # delete the other samples 
    del labels_dict
    
    print('Creating dataset with total classess of {} ..'.format(len(labels_dict_train)))
        
    train_x, train_y = create_pair(labels_dict_train, labels, True)

    print('Total samples created for training: {}'.format(len(train_y)))
    
    test_x, test_y = create_pair(labels_dict_test, labels, False)
    
    print('Total samples created for testing: {}'.format(len(test_y)))
    
    return train_x, test_x, train_y, test_y, labels_dict_train, labels_dict_test, zero_shot_set, zero_shot_test_samples, zero_shot_test_labels

train_x, test_x, train_y, test_y = 0, 0, 0, 0

train_paired, test_paired, \
train_paired_target, test_paired_target, \
labels_dict_train, labels_dict_test, \
zero_shot_set, zero_shot_test_samples, zero_shot_test_labels = create_pairwise_dataset(x, y)

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
  
feature_size = pair1_train.shape[-1]
pair1_train = pair1_train.reshape(-1, feature_size)
pair2_train = pair2_train.reshape(-1, feature_size)  
pair1_test = pair1_test.reshape(-1, feature_size)
pair2_test = pair2_test.reshape(-1, feature_size)   

# distance metrics
def cosine_distance(z):
    x = K.l2_normalize(z[0])
    y = K.l2_normalize(z[1])
    return K.batch_dot(x, y, axes=-1)

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1, keepdims=True)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1, keepdims=True)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# siamese neural network structure 
input_shape = pair1_train.shape[1]
left_input = Input(shape=(input_shape,))
right_input = Input(shape=(input_shape,))

# 60%
#nnet = Sequential()
#nnet.add(Dense(100, activation='relu', input_dim=input_shape))
##nnet.add(Dense(500, activation='relu'))
##nnet.add(Dense(100, activation='relu'))
#nnet.add(Dense(50, activation='relu'))

# 70%
#nnet = Sequential()
#nnet.add(Dense(500, activation='relu', input_dim=input_shape))
#nnet.add(Dense(250, activation='relu'))
##nnet.add(Dense(100, activation='relu'))
#nnet.add(Dense(50, activation='relu'))

#nnet = Sequential()
#nnet.add(Dense(1000, activation='relu', input_dim=input_shape))
#nnet.add(Dense(500, activation='relu'))
#nnet.add(Dense(100, activation='relu'))
#nnet.add(Dense(50, activation='relu'))

kernel_size = 4
nnet = Sequential()
#nnet.add(Dense(1000, kernel_regularizer=l1(1e-3), activation='relu'))
#nnet.add(Dropout(0.2))
#nnet.add(Dense(500, kernel_regularizer=l1(1e-3), activation='relu'))
#nnet.add(Dropout(0.2))
nnet.add(Dense(100, kernel_regularizer=l2(0.01), activation='relu'))
#nnet.add(Dropout(0.2))
#nnet.add(Dense(10, kernel_regularizer=l1(1e-2), activation='relu'))

l2_penalization = {}
l2_penalization['Conv1'] = 1e-2
l2_penalization['Conv2'] = 1e-2
l2_penalization['Conv3'] = 1e-2
l2_penalization['Conv4'] = 1e-2
l2_penalization['Dense1'] = 1e-2
    
# convolutional net
convolutional_net = Sequential()
convolutional_net.add(Reshape((-1, 1), input_shape=(input_shape,)))
convolutional_net.add(Conv1D(filters=8, kernel_size=50,
                             activation='relu',
                             kernel_regularizer=l1(
                                 l2_penalization['Conv1']),
                             name='Conv1'))                            
convolutional_net.add(MaxPooling1D(2))

convolutional_net.add(Conv1D(filters=8, kernel_size=50,
                             activation='relu',
                             kernel_regularizer=l2(
                                 l2_penalization['Conv2']),
                             name='Conv2'))
convolutional_net.add(MaxPooling1D(2))

#convolutional_net.add(Conv1D(filters=8, kernel_size=4,
#                             activation='relu',
#                             kernel_regularizer=l2(
#                                 l2_penalization['Conv3']),
#                             name='Conv3'))
#convolutional_net.add(MaxPooling1D(2))
#
#convolutional_net.add(Conv1D(filters=16, kernel_size=2,
#                             activation='relu',
#                             kernel_regularizer=l1(
#                                 l2_penalization['Conv4']),
#                             name='Conv4'))

convolutional_net.add(Flatten())
convolutional_net.add(
    Dense(units=10, activation='relu',
          kernel_regularizer=l1(
              l2_penalization['Dense1']),
          name='Dense1'))
                  
                  
#encode each of the two inputs into a vector with the convnet
encoded_l = nnet(left_input)
encoded_r = nnet(right_input)

#merge two encoded inputs with the l1 distance between them
distance = lambda x: K.abs(x[0] - x[1])
merge = Lambda(distance)([encoded_l, encoded_r])
merge = Dense(1, activation='sigmoid')(merge)
siamese_net = Model(input=[left_input, right_input], output=merge)
siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(siamese_net.summary())

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5, min_delta=0, restore_best_weights=True)

siamese_net.fit([pair1_train, pair2_train], train_paired_target, 
                validation_data=([pair1_test, pair2_test], test_paired_target),
                epochs=200, batch_size=500, shuffle=True, callbacks=[es])

#siamese_net.save('siamese.h5')
#siamese_net = load_model('siamese.h5')

#================== evaluation for siamese network ========================#

# get the support vectors set
#support_set = {}
#counter = 1
#for labelA in sorted(labels_dict_train):
#    if counter > limit:
#        break
#    else:
#        support_set[labelA] = labels_dict_train[labelA]
#        counter += 1
#
## get samples to be tested
#test_samples = []
#test_labels = []
#counter = 1
#for labelA in sorted(labels_dict_test):
#    if counter > limit:
#        break
#    else:
#        samples = labels_dict_test[labelA]
#        idxs = np.random.permutation(len(samples))[:3]
#        for k in idxs:
#            test_samples.append(samples[k])
#            test_labels.append(labelA)
#        counter += 1  

# run testing in a non-parametric way    
#correct = 0
#total = 0
#for i in range(len(test_samples)):
#    outputs = []
#    labels = []
#    for s in support_set:
#        samples = support_set[s]
#        idx = np.random.permutation(len(samples))[:5]
#        d = 0
#        for j in idx:
#            d += siamese_net.predict([samples[j], test_samples[i]])
#        distance = d / len(idx)
##        print('True label: {}, Label: {}, Distance: {}'.format(test_labels[i], s, distance))
#        outputs.append(distance)
#        labels.append(s)
#    topK = topk_var
#    idxs = sorted(range(len(outputs)), key=lambda z: outputs[z])[:topK]
#    possible_labels = []
#    for z in idxs:
#        possible_labels.append(labels[z])
#    if test_labels[i] in possible_labels:
#        correct += 1
#    total += 1
#print('Accuracy on {} for top {}: {}%'.format(limit, topk_var, (correct/total)*100))

# =========================== Zero-Shot testing =============================#

# run testing in a non-parametric way   
combined_support_set = {**zero_shot_set}#, **support_set} 
correct = 0
total = 0
for i in range(len(zero_shot_test_samples)):
    outputs = []
    labels = []
    for s in combined_support_set:
        samples = combined_support_set[s]
        idx = [3] #np.random.permutation(len(samples))[:3]
        d = 0
        for j in idx:
            d += siamese_net.predict([zero_shot_test_samples[i], samples[j]])
        distance = d / len(idx)
        outputs.append(distance)
        labels.append(s)
    topK = topk_var
    idxs = sorted(range(len(outputs)), key=lambda z: outputs[z])[:topK]
    possible_labels = []
    for z in idxs:
        possible_labels.append(labels[z])
    if zero_shot_test_labels[i] in possible_labels:
        correct += 1
    total += 1

print(total)
print(correct)
print('Accuracy for one-shot learning on {} for top {}: {}%'.format(limit, topk_var, (correct/total)*100))


#from sklearn.neighbors import KNeighborsClassifier
#X = []
#y= []
#counter = 1
#for label in sorted(labels_dict_train):
#    if counter > limit:
#        break
#    samples = labels_dict_train[label]
#    for s in samples:
#        X.append(s)
#        y.append(label)
#    counter += 1
#X = np.array(X)
#X = X.reshape(-1, X.shape[-1])
#
#X_test = []
#y_test = []
#counter = 1
#for label in sorted(labels_dict_test):
#    if counter > limit:
#        break
#    samples = labels_dict_test[label]
#    for s in samples:
#        X_test.append(s)
#        y_test.append(label)
#    counter += 1
#X_test = np.array(X_test)
#X_test = X_test.reshape(-1, X_test.shape[-1])  
#
#
#neigh = KNeighborsClassifier(n_neighbors=10)
#neigh.fit(X, y)
#print('KNN accuracy: ', neigh.score(X_test, y_test))
