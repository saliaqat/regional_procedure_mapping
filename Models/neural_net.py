import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
from Models.model import Model
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D, GRU, RNN
from keras.layers import Input
from keras.layers import Dense
from keras.layers import RNN, SimpleRNN
from keras.models import Model as kerasModel
from keras.utils import to_categorical
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder

class NeuralNet(Model):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
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
        # x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data(self, x, y):
        # x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.test_x = x
        self.test_y = y

    def set_train_data_np(self, x, y):
        # x = x.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data_np(self, x, y):
        # x = x.reshape(x.shape[0], 1, x.shape[1])
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
        NeuralNet.__init__(self, in_shape, regional_labels, batch_size, epochs, batch_size=batch_size)

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
        self.model.compile(optimizer="adam", loss='categorial_crossentropy', metrics=["acc"])
        self.name = 'multiclassSimpleCNN'

class MultiClassSimpleNN(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
        NeuralNet.__init__(self, in_shape, regional_labels, epochs=epochs, batch_size=batch_size)

        features = in_shape[1]

        inputs = Input(shape=(features, ), name="input")

        x = Dense(4096, activation="relu", name="dense1")(inputs)
        x = Dense(4096, activation="relu", name="dense2")(x)
        x = Dense(4096, activation="relu", name="dense3")(x)
        x = Dense(4096, activation="relu", name="dense4")(x)
        x = Dense(2048, activation="relu", name="dense5")(x)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='categorial_crossentropy', metrics=["acc"])
        self.name = 'multiclassSimpleNN'

class MultiClassNN(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
        NeuralNet.__init__(self, in_shape, regional_labels, epochs=epochs, batch_size=batch_size)

        features = in_shape[1]

        inputs = Input(shape=(features, ), name="input")

        # x = Dense(2048, activation="relu", name="dense1")(inputs)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(inputs)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='categorial_crossentropy', metrics=["acc"])
        self.name = 'multiclassNN'

class MultiClassNNBig(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
        NeuralNet.__init__(self, in_shape, regional_labels, epochs=epochs, batch_size=batch_size)

        features = in_shape[1]

        inputs = Input(shape=(features, ), name="input")

        x = Dense(8192, activation="relu", name="dense1")(inputs)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='categorial_crossentropy', metrics=["acc"])
        self.name = 'multiclassNN'

class MultiClassNNScratch(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
        NeuralNet.__init__(self, in_shape, regional_labels, epochs=epochs, batch_size=batch_size)

        features = in_shape[1]

        inputs = Input(shape=(features, ), name="input")

        x = Dense(8192, activation="relu", name="dense1")(inputs)
        x = Dense(4096, activation="relu", name="dense2")(x)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.summary
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["acc", top_3_accuracy, top_k_categorical_accuracy])
        self.name = 'multiclassNNScratch'

    def set_train_data(self, x, y):
        # x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data(self, x, y):
        # x = x.A.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.test_x = x
        self.test_y = y

    def train(self, val_x, val_y):
        # val_x =val_x.A.reshape(val_x.shape[0], 1, val_x.shape[1])
        callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                     ModelCheckpoint(filepath='best_nn_model_%s.h5' % self.name, monitor='val_loss',
                                     save_best_only=True)]
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs,
                       callbacks=callbacks, validation_data=(val_x, self.encoder.transform(val_y)))

class MultiClassNNScratchAuto(NeuralNet):
    def __init__(self, in_shape, regional_labels, batch_size=64, epochs=50):
        NeuralNet.__init__(self, in_shape, regional_labels, epochs=epochs, batch_size=batch_size)

        features = in_shape[1]

        inputs = Input(shape=(1, features), name="input")

        x = SimpleRNN(4096, activation="relu", name="RNN1")(inputs)
        y = Dense(self.encoder.transform(self.labels).shape[1], activation="softmax", name="output")(x)

        self.model = kerasModel(inputs, y)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["acc", top_3_accuracy, top_k_categorical_accuracy])
        self.name = 'multiclassNNScratch'

    def set_train_data(self, x, y):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.train_x = x
        self.train_y = y

    def set_test_data(self, x, y):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        y = self.encoder.transform(y)

        self.test_x = x
        self.test_y = y

    def train(self, val_x, val_y):
        val_x =val_x.reshape(val_x.shape[0], 1, val_x.shape[1])
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     ModelCheckpoint(filepath='best_nn_model_%s.h5' % self.name, monitor='val_loss',
                                     save_best_only=True)]
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs,
                       callbacks=callbacks, validation_data=(val_x, self.encoder.transform(val_y)))


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)