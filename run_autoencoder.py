from keras.layers import Input, Dense
from keras.models import Model, load_model
from data_reader import DataReader
from data_manipulator import *
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D

from keras.optimizers import RMSprop
from sklearn.linear_model import LogisticRegression
from Models.logistic_regression import *
from keras.models import Model

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def run_autoencoder(train_x, train_y, test_x, test_y, dim):
    
    # run autoencoder
    vocab_size = train_x.shape[1]
    
    # this is the size of encoded representations
    encoding_dim = dim 

    # this is  input placeholder
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

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.fit(train_x, train_x,
                epochs=10,
                batch_size=250,
                shuffle=True,
                validation_data=(test_x, test_x))

    encoder.save('encoder-' + str(dim) + '.h5')
#    encoder = load_model('encoder.h5')
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)

    from IPython import embed
    embed()
    
    lg = LogisticRegression(penalty='l2', solver='newton-cg', n_jobs=1)
    lg.fit(encoded_train, train_y)
    print('Accuracy for ' + str(dim) + 'dimensions: ', lg.score(encoded_test, test_y))


# update: Logistic Regression after dimension reduction to 1000: 79.5%
def main():
    # get data
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
    
    tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_short=True, remove_repeats=True)
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)
    
    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False, remove_short=True, remove_repeats=True)
    test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, feature_names=feature_names)
    
    print(train_x.shape)
    print(test_x.shape)
    
    dims = [10, 500, 1000, 2500]
    for dim in dims:
        if dim > 1002:
            run_autoencoder(train_x, train_y, test_x, test_y, dim)
        

    
if __name__ == '__main__':
    main()
