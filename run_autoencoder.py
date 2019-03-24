from keras.layers import Input, Dense
from keras.models import Model, load_model
from data_reader import DataReader
from data_manipulator import *
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from sklearn import preprocessing
from keras.optimizers import RMSprop

# from cluster_by_site import Cluster
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from Models.logistic_regression import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def run_autoencoder(train_x_raw, train_x, train_y, test_x, test_y, feature_names):

    print(train_x_raw[:10])
    # run autoencoder
    vocab_size = len(feature_names)
    # this is the size of our encoded representations
    encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


    # this is  input placeholder
    input_sequence = Input(shape=(vocab_size,))
    
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_sequence)
    
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(vocab_size, activation='softmax')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_sequence, decoded)
    autoencoder.summary()

    # this model maps an input to its encoded representation
    encoder = Model(input_sequence, encoded)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    autoencoder.fit(train_x, train_x,
                epochs=100,
                batch_size=250,
                shuffle=True,
                validation_data=(test_x, test_x))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_sentences = encoder.predict(train_x)
    decoded_sentences = decoder.predict(encoded_sentences)
    #print(train_x[0])
    #print(np.where(decoded_sentences[0] == max(decoded_sentences[0])))
    top_5_idx = np.argsort(decoded_sentences[0])[-5:]
    top_5_values = [decoded_sentences[0][i] for i in top_5_idx]
    #print(top_5_idx)
    #print(decoded_sentences[0])
   #    print(decoded_sentences[0][np.where(decoded_sentences[0] == max(decoded_sentences[0]))])
    # run clustering
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(encoded_sentences[:500])
    labels = kmeans.labels_
    cluster = Cluster(kmeans, None, encoded_sentences, labels)

    # run a query
    regex_string=r'[a-zA-Z0-9]+'
    tokenizer = RegexpTokenizer(regex_string)

    # get representation (bag of words)
    query = "Y DIR - ANGIOGRAM"
    tokens = tokenizer.tokenize(query.lower())
    weights, y, _ = tokens_to_bagofwords([tokens, ], 1, CountVectorizer, feature_names)
    encoded_test_x = encoder.predict(weights)
    neighbours, neighbours_idx = cluster.get_nearest_neighbours(encoded_test_x)
    for n in neighbours:
        decoded_neighbour = decoder.predict(n.reshape(1,encoding_dim))
        print(decoded_neighbour.shape)
        top_5_idx = np.argsort(decoded_neighbour)[0][-5:]
        top_5_values = [decoded_neighbour[0][i] for i in top_5_idx]
        print(top_5_idx)
        bow = ""
        for i in top_5_idx:
            bow += feature_names[i] + " "
        print(bow)
        print(decoder.predict(n.reshape(1,encoding_dim)).shape)
    exit(0)
    #print(neighbours_idx)
    #print(train_x[neighbours_idx])
    for n in neighbours_idx[0]:
        print(n)
        feature_idx = np.where(train_x[n].toarray()[0] > 0)[0]
        print(feature_idx)
        if len(feature_idx) > 0 :
            for i in feature_idx:
                print(feature_names[i])
        print(train_x[n].shape)
        print(type(train_x[n]))
    #print(neighbours_idx)
    #cluster.get_nearest_neighbours("Y DIR - ANGIOGRAM")



# update: Logistic Regression after dimension reduction to 1000: 79.5%
def main():
    # get data
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
    train_y_raw = train_x_raw['RIS PROCEDURE CODE']
    test_y_raw = test_x_raw['RIS PROCEDURE CODE']
    train_x_raw = train_x_raw.drop("RIS PROCEDURE CODE", axis=1)
    test_x_raw = test_x_raw.drop("RIS PROCEDURE CODE", axis=1)

    # tokenize and bag of words it
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, False, r'[a-zA-Z0-9]+',True, True)
    print(tokens.shape)
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)
    tokens1, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, False , r'[a-zA-Z0-9]+',True, True)
    test_x, test_y, _ = tokens_to_bagofwords(tokens1, train_y_raw, CountVectorizer, feature_names)
    print(train_x.shape)
    # start autoencoder
    run_autoencoder(train_x_raw, train_x, train_y, test_x, test_y, feature_names)



def get_encoder(train_x, test_x, dim):
    # run autoencoder
    vocab_size = train_x.shape[1]

    # this is the size of encoded representations
    encoding_dim = dim

    # this is  input placeholder
    input_sequence = Input(shape=(vocab_size,))
    print(train_x.shape)

    layer_one_size = int(encoding_dim + (2 * ((vocab_size - encoding_dim) / 3)))
    layer_two_size = int(encoding_dim + (1 * ((vocab_size - encoding_dim) / 3)))

    other_layer_size = int(encoding_dim + (1 * ((vocab_size - encoding_dim) /8)))

    # "encoded" is the encoded representation of the input
    x = Dense(other_layer_size, activation='relu', name='enc_dense_1')(input_sequence)
    # x = Dense(layer_one_size, activation='relu', name='enc_dense_1')(input_sequence)
    # x = Dense(layer_two_size, activation='relu', name='enc_dense_2')(x)

    encoded = Dense(encoding_dim, activation='relu', name='encoded')(x)

    y = Dense(other_layer_size, activation='relu', name='dec_dense_1')(encoded)
    # y = Dense(layer_two_size, activation='relu', name='dec_dense_1')(encoded)
    # y = Dense(layer_one_size, activation='relu', name='dec_dense_2')(y)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(vocab_size, activation='sigmoid', name='output')(y)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_sequence, outputs=decoded)
    autoencoder.summary()

    # this model maps an input to its encoded representation
    encoder = Model(input_sequence, encoded)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    # decoder = Model(encoded_input, decoder_layer)
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath='best_autoencoder.h5', monitor='val_loss',
                                 save_best_only=True)]

    autoencoder.fit(train_x, train_x,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(test_x, test_x), callbacks=callbacks)

    return encoder

if __name__ == '__main__':
    main()
