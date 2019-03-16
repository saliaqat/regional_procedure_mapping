from __future__ import print_function, division
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")


n_epochs = 20
n_episodes = 100
n_way = 60
n_shot = 5
n_query = 5
n_examples = 20
inp_width, inp_dim = 200, 1
h_dim = 64
z_dim = 64

def euclidean_distance(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=-1))    

x = Input(shape=(200, 1))

x = Conv1D(h_dim, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D()(x)

x = Conv1D(h_dim, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D()(x)

x = Conv1D(h_dim, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D()(x)

x = Conv1D(z_dim, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D()(x)

#x = Flatten()(x)
