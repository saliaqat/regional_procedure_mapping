import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
from Models.model import Model
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D, GRU, RNN
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model as kerasModel
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

class NeuralNet(Model):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=1):
        self.batch_size = batch_size
        self.epochs = epochs

        self.labels = regional_labels
        self.set_one_hot_encoder()
        self.name = 'multiclassSimpleCNN'
        pass

    def set_one_hot_encoder(self):
        encoder = OneHotEncoder()
        self.labels = self.labels.reshape(self.labels.shape[0], 1)
        encoder.fit(self.labels)
        self.encoder = encoder

    def set_train_data(self, x, y):
        x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data(self, x, y):
        x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.test_x = x
        self.test_y = y

    def set_train_data_np(self, x, y):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data_np(self, x, y):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.test_x = x
        self.test_y = y

    def train(self):
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self):
        self.predict_y = self.model.predict(self.test_x)
        return self.predict_y

    # def predict(self, x):
    #     y = self.model.predict(x)
    #     return y

    def score(self):
        self.the_score = accuracy_score(np.argmax(self.test_y, axis=1), np.argmax(self.predict_y, axis=1))
        # self.score = self.model.evaluate(self.test_x, self.test_y)
        return self.the_score

class MultiClassSimpleCNN(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=1):
        NeuralNet.__init__(self, in_shape, regional_labels, batch_size, epochs)

        features = in_shape[1]

        inputs = Input(shape=(1, features), name="input")

        x = Conv1D(16, 20, padding="same", activation="relu", name="conv1")(inputs)
        x = MaxPooling1D(5, padding="same", name="pool1")(x)

        x = Conv1D(16, 10, padding="same", activation="relu", name="conv2")(x)
        x = MaxPooling1D(5, padding="same", name="pool2")(x)

        x = Flatten()(x)
        x = Dropout(0.2, name="drop1")(x)

        x = Dense(256, activation="relu", name="dense1")(x)
        x = Dropout(0.2, name="drop2")(x)

        x = Dense(128, activation="relu", name="dense2")(x)
        x = Dropout(0.2, name="drop3")(x)

        x = Dense(64, activation="relu", name="dense3")(x)
        x = Dropout(0.2, name="drop4")(x)

        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
        self.name = 'multiclassSimpleCNN'

class MultiClassSimpleNN(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=5):
        NeuralNet.__init__(self, in_shape, regional_labels)

        features = in_shape[1]

        inputs = Input(shape=(1, features), name="input")

        x = Dense(2048, activation="relu", name="dense1")(inputs)
        x = Dense(3072, activation="relu", name="dense2")(x)
        x = Dense(4096, activation="relu", name="dense3")(x)
        x = Dense(3072, activation="relu", name="dense4")(x)
        x = Dense(2048, activation="relu", name="dense5")(x)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
        self.name = 'multiclassSimpleNN'