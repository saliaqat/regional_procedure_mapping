from __future__ import print_function, division
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.sparse import vstack
from sklearn import preprocessing
from data_reader import DataReader
from data_manipulator import *
import warnings
warnings.filterwarnings("ignore")


# ================= Prepare the data for Proto Net ===========================#
# get data
data_reader = DataReader()
df = data_reader.get_all_data()
train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_empty=True, remove_repeats=True, remove_short=True)
train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw, vectorizer_class=CountVectorizer)

tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False, remove_empty=True, remove_repeats=True, remove_short=True)
test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, vectorizer_class=CountVectorizer, feature_names=feature_names)

# encode the labels into smaller integers rather than large integers
le = preprocessing.LabelEncoder()
le.fit(np.concatenate((test_y.values, train_y.values)))
train_y = le.transform(train_y.values)
test_y = le.transform(test_y.values)

# combine train and test
x = vstack((train_x, test_x)).todense()
y = np.concatenate((train_y, test_y))

feature_length = x.shape[1]
limit = 100

# create dataset for siamese neural network
# k is the minimum sample needed for each class
def create_dataset(x, y, k=15, train_ratio=0.66, limit=100):
        
    labels_dict = {}
    for i in range(len(y)):
        if y[i] in labels_dict:
            # cap at only k samples per class
            if len(labels_dict[y[i]]) <= k:
                labels_dict[y[i]].append(x[i].reshape(feature_length, 1))
        else:
            labels_dict[y[i]] = [x[i].reshape(feature_length, 1)]

    labels_dict_train = {}
    labels_dict_test = {}
    labels = []
    counter = 1
    for label in sorted(labels_dict):
        if counter > limit:
            break
        if len(labels_dict[label]) >= k:
            labels_dict_train[label] = labels_dict[label][:int(np.ceil(k*train_ratio))]
            labels_dict_test[label] = labels_dict[label][int(np.ceil(k*train_ratio)):k]
            labels.append(label)
        counter += 1
    
    # create training dataset
    train_dataset = []
    train_labels = []
    for label in labels_dict_train:
        samples = labels_dict_train[label]
        samples = np.hstack(samples)
        train_dataset.append(samples)
        train_labels.append(label)
        
    # create testing dataset
    test_dataset = []
    test_labels = []
    for label in labels_dict_test:
        samples = labels_dict_test[label]
        samples = np.hstack(samples)
        test_dataset.append(samples)
        test_labels.append(label)
    
    # reshape train dataset to (num_class, num_samples, features)
    train_dataset = np.array(train_dataset)
    train_dataset = train_dataset.reshape(train_dataset.shape[0], train_dataset.shape[2], train_dataset.shape[1])
    test_dataset = np.array(test_dataset)
    test_dataset = test_dataset.reshape(test_dataset.shape[0], test_dataset.shape[2], test_dataset.shape[1])
    
    return train_dataset, test_dataset, np.array(train_labels), np.array(test_labels)

train_dataset, test_dataset, train_labels, test_labels = create_dataset(x, y, k=15, train_ratio=0.66, limit=limit)

train_x, test_x, train_y, test_y = 0, 0, 0, 0

# encode the labels into smaller integers rather than large integers
le = preprocessing.LabelEncoder()
le.fit(np.concatenate((train_labels, test_labels)))
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)
train_onehot_labels = np_utils.to_categorical(train_labels)
test_onehot_labels = np_utils.to_categorical(test_labels)

# ================= Create Prototypical Netowrks model ======================#
n_c = limit # total np. of classes
n_epi_c = 100 # no. of classes per training episode
n_epochs = 10000000000
n_episodes = int(n_c / n_epi_c)
n_support = 5
n_query = 5

def euclidean_distance(x):
    return K.mean(K.abs(x[0] - x[1]), axis=-1, keepdims=True)  

def cosine_distance(x):
    a, b = x[0], x[1]
    a = K.l2_normalize(a, axis=-1)
    b = K.l2_normalize(b, axis=-1)
    return -K.mean(a * b, axis=-1, keepdims=True)
  
# define the encoder
encoder = Sequential()
encoder.add(Dense(1000, activation='relu', input_dim=feature_length))
encoder.add(Dense(500, activation='relu'))
encoder.add(Dense(250, activation='relu'))

#encode each of the two inputs into a vector with the convnet
support_inp = Input(shape=(n_support, feature_length))
query_inp = Input(shape=(n_query, feature_length))

encoded_support = encoder(support_inp)
encoded_support =  Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(encoded_support)
encoded_query = encoder(query_inp)
distance = Lambda(euclidean_distance)([encoded_support, encoded_query])
softmax = Dense(n_c, activation='softmax')(distance)

proto_net = Model(input=[support_inp, query_inp], output=softmax)
opt = Adam(lr=0.1)
proto_net.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
print(proto_net.summary())

for e in range(1, n_epochs + 1):
    
    for i in range(1, n_episodes + 1):
        
        # randomly sample classes and corresponding data for this training episode
        idx_class = np.random.permutation(n_c)[:n_epi_c]
        epi_data = train_dataset[idx_class]
        epi_label = train_onehot_labels[idx_class]
        epi_label = np.tile(epi_label, n_query).reshape(-1, n_query, n_c)

        #randomlpy sample the query and support set
        idxs = np.random.permutation(n_support + n_query)
        idx_support = idxs[:n_support]
        idx_query = idxs[n_support:]
        support = epi_data[:, idx_support, :]
        query = epi_data[:, idx_query, :]
        loss = proto_net.train_on_batch(x=[support, query], y=epi_label)
    
    # update the learning rate every 200 epochs
    if e % 200 == 0:
        curr_lr = K.eval(proto_net.optimizer.lr) * 0.8
        K.set_value(proto_net.optimizer.lr, curr_lr)
        
    print('{}: Training loss: {}, Training acc: {}'.format(e, round(loss[0], 4), round(loss[1], 4)))
