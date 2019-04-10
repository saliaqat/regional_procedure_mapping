from __future__ import print_function, division
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import numpy as np
from sklearn import preprocessing
from data_reader import DataReader
from data_manipulator import *
from Model.siamese import SiameseNN
from Model.knn import KNN
import sys
import random
import warnings
warnings.filterwarnings("ignore")
    
# helper function to shuffle data a, b, and c in unison
def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

# helper function to create pairs for the pairwise dataset
def create_pair(labels_dict, labels):
    
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

    
    while len(targets_diff) != int(len(targets_sim)):
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

# helper function to create pairwise dataset for siamese neural network
def create_pairwise_dataset(x, y, k=15, train_ratio=0.66, limit=100):
        
    labels_dict = {}
    for i in range(len(y)):
        if y[i] in labels_dict:
            if len(labels_dict[y[i]]) <= k:
                labels_dict[y[i]].append(x[i])
        else:
            labels_dict[y[i]] = [x[i]]

    labels_dict_train = {}
    labels_dict_test = {}
    labels = []
    uc_test_samples = [] # uc = unknown class
    uc_test_labels = []
    uc_support_set = {}
    counter = 0
    m = 0
    
    # randomly pick unknown classes to test the model on unseen classes
    unknown_class_indices = np.random.permutation(np.arange(limit+1, 1000))[:zero_shot]

    for label in sorted(labels_dict):
        if counter > limit + zero_shot:
            break
        elif counter > limit:
            if len(labels_dict[unknown_class_indices[m]]) >= k:
                uc_support_set[unknown_class_indices[m]] = labels_dict[unknown_class_indices[m]][:int(np.ceil(k*train_ratio))]
                samples = labels_dict[unknown_class_indices[m]][int(np.ceil(k*train_ratio)):k]
                for s in samples:
                    uc_test_samples.append(s)
                    uc_test_labels.append(unknown_class_indices[m])
                m += 1
        else:
            if len(labels_dict[label]) >= k: 
                labels_dict_train[label] = labels_dict[label][:int(np.ceil(k*train_ratio))]
                labels.append(label)
        counter += 1
        
    counter = 0
    test_limit = int(limit)
    for label in sorted(labels_dict):
        if counter > test_limit:
            break
        if (label in labels_dict_train) or (label in uc_support_set):
            pass
        else:
            labels_dict_test[label] = labels_dict[label][:int(np.ceil(k*train_ratio))]
            counter += 1
            
    # delete the other samples 
    del labels_dict
    
    print('Creating dataset with total classess of {} ..'.format(len(labels_dict_train)))
        
    train_x, train_y = create_pair(labels_dict_train, labels)

    print('Total samples created for training: {}'.format(len(train_y)))
    
    test_x, test_y = create_pair(labels_dict_test, labels)
    
    print('Total samples created for testing: {}'.format(len(test_y)))
    
    return train_x, test_x, train_y, test_y, labels_dict_train, labels_dict_test, uc_support_set, uc_test_samples, uc_test_labels


# function to run few shot algorithm with Siamese NN
def siamese_fewshot(train, snn_seen_score, snn_unseen_score, knn_score, nn_score):
    
    # get the data
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
    y = np.concatenate((train_y, test_y))
    
    # delete unwanted variables        
    del train_x, test_x, train_y, test_y
    
    # create pairwise dataset
    train_paired, test_paired, \
    train_paired_target, test_paired_target, \
    labels_dict_train, labels_dict_test, \
    uc_support_set, uc_test_samples, uc_test_labels = create_pairwise_dataset(x, y)
    
    # set the data as separate numpy array to be passed into model
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
    
    # if train the model from scratch
    if train:
    
        # siamese neural network structure 
        input_shape = pair1_train.shape[1]

        siamese_net = SiameseNN(input_shape)

        siamese_net.train([pair1_train, pair2_train], train_paired_target, 
                          [pair1_test, pair2_test], test_paired_target)
        
        siamese_net.save('siamese-' + str(limit) + '.h5')
     
    # if load the pre-trained model
    else:
        
        siamese_net = SiameseNN(input_shape)
        siamese_net.load_model('siamese-' + str(limit) + '.h5')
    
    #========================================================================#
    #======================= Prepare data for testing ========================#
    #========================================================================#
    
    # support set for SNN testing
    support_set = {}
    counter = 0
    for labelA in sorted(labels_dict_train):
        if counter > limit:
            break
        else:
            support_set[labelA] = labels_dict_train[labelA]
            counter += 1
        
    # training data for kNN and neural network
    x_train = []
    y_train = []
    counter = 1
    for label in sorted(labels_dict_train):
        if counter > limit:
            break
        samples = labels_dict_train[label]
        for s in samples:
            x_train.append(s)
            y_train.append(label)
        counter += 1
    x_train = np.array(x_train)
    x_train = x_train.reshape(-1, x_train.shape[-1])
    y_train = np.array(y_train).reshape(-1, 1)
    
    # testing data for kNN and neural network
    test_samples = []
    test_labels = []
    counter = 0
    for labelA in sorted(labels_dict_test):
        if counter > limit:
            break
        else:
            samples = labels_dict_test[labelA]
            idxs = np.random.permutation(len(samples))[:3]
            for k in idxs:
                test_samples.append(samples[k])
                test_labels.append(labelA)
            counter += 1  
    test_samples = np.array(test_samples)
    test_samples = test_samples.reshape(-1, test_samples.shape[-1])
    test_labels = np.array(test_labels).reshape(-1, 1)
    
    #========================================================================#
    #================ Evaluation for SNN on seen classes ====================#
    #========================================================================#
    
    if snn_seen_score:
        
        # run testing in a non-parametric way    
        correct = 0
        total = 0
        for i in range(len(test_samples)):
            outputs = []
            labels = []
            for s in support_set:
                samples = support_set[s]
                idx = np.random.permutation(len(samples))
                d = 0
                for j in idx:
                    d += siamese_net.predict([samples[j], test_samples[i]])
                distance = d / len(idx)
                outputs.append(distance)
                labels.append(s)
            topK = topk_var
            idxs = sorted(range(len(outputs)), key=lambda z: outputs[z])[:topK]
            possible_labels = []
            for z in idxs:
                possible_labels.append(labels[z])
            if test_labels[i] in possible_labels:
                correct += 1
            total += 1
           
        score = (correct/total)*100
        score = (correct/total)*100
        print('SNN seen class score: {}%: '.format(score))

    #========================================================================#
    #=================== Evaluation with baseline KNN =======================#
    #========================================================================#
        
    if knn_score:
        
        knn = KNN(n_neighbors=10)
        knn.train(x_train, y_train)
        score = knn.score(test_samples, test_labels)*100
        print('kNN score: {}%: '.format(score))
    
    #========================================================================#
    #=================== Evaluation with Neural Network =====================#
    #========================================================================#
    
    if nn_score:
        
        inputs = Input(shape=(feature_size, ), name="input")
    
        x = Dense(7750, activation="relu", name="dense1", input_dim=feature_size)(inputs)
        output = Dense(limit, activation="softmax", name="output")(x)
        nn = Model(inputs, output)    
        nn.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)]
        
        y_train_onehot = np_utils.to_categorical(y_train)
        
        test_labels_onehot = np_utils.to_categorical(test_labels)
        nn.fit(x_train, y_train_onehot, batch_size=256, epochs=100,
                       callbacks=callbacks, validation_data=(test_samples, test_labels_onehot))
        
        score, acc = nn.evaluate(test_samples, test_labels_onehot)
        
        print('Neural network score: {}%: '.format(score))
    
    #========================================================================#
    #============== Evaluation for SNN on unseen classes ====================#
    #========================================================================#
    
    if snn_unseen_score:
        
        # run testing in a non-parametric way   
        correct = 0
        total = 0
        for i in range(len(uc_test_samples)):
            outputs = []
            labels = []
            for s in uc_support_set:
                samples = uc_support_set[s]
                idx = np.random.permutation(len(samples))
                d = 0
                for j in idx:
                    d += siamese_net.predict([uc_test_samples[i], samples[j]])
                distance = d / len(idx)
                outputs.append(distance)
                labels.append(s)
            topK = topk_var
            idxs = sorted(range(len(outputs)), key=lambda z: outputs[z])[:topK]
            possible_labels = []
            for z in idxs:
                possible_labels.append(labels[z])
            if uc_test_labels[i] in possible_labels:
                correct += 1
            total += 1
            
        score = (correct/total)*100
        print('SNN unseen class score: {}%: '.format(score))


if __name__ == '__main__':

    variable1 = int(sys.argv[1])
    limit = variable1
    variable2 = int(sys.argv[2])
    zero_shot = variable2

    siamese_fewshot(train=True, 
                    snn_seen_score=True, 
                    snn_unseen_score=True, 
                    knn_score=True, 
                    nn_score=True)
