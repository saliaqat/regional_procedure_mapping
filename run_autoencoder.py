from keras.layers import Input, Dense
from keras.models import Model
from data_reader import DataReader
from data_manipulator import *
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def run_autoencoder(train_x, train_y, test_x, test_y, feature_names):
    # run autoencoder
    vocab_size = len(feature_names)
    print(train_x[0].shape)
    # this is the size of our encoded representations
    encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_sequence = Input(shape=(vocab_size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_sequence)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(vocab_size, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_sequence, decoded)
    autoencoder.summary()

    # this model maps an input to its encoded representation
    encoder = Model(input_sequence, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.fit(train_x, train_x,
                epochs=5,
                batch_size=250,
                shuffle=True,
                validation_data=(test_x, test_x))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_sentences = encoder.predict(test_x)
    decoded_sentences = decoder.predict(encoded_sentences)

    print(encoded_sentences.shape)

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
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)
    tokens1, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y, _ = tokens_to_bagofwords(tokens1, train_y_raw, CountVectorizer, feature_names)

    # start autoencoder
    run_autoencoder(train_x, train_y, test_x, test_y, feature_names)

if __name__ == '__main__':
    main()
