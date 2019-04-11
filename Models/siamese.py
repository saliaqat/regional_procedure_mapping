import numpy as np
from Models.model import Model
from keras.layers import Input, Dense, Flatten, Dropout, Add, Lambda, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model as KerasModel, load_model
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1
import keras.backend as K

class SiameseNN(Model):
    def __init__(self, input_shape, reg_lambda=0.00):

        left_input = Input(shape=(input_shape,))
        right_input = Input(shape=(input_shape,))
                
        nnet = Sequential()
        nnet.add(Dense(1000, kernel_regularizer=l2(reg_lambda), activation='relu'))
        nnet.add(Dense(500, kernel_regularizer=l2(reg_lambda), activation='relu'))
        nnet.add(Dense(250, kernel_regularizer=l2(reg_lambda), activation='relu'))
        nnet.add(Dense(100, kernel_regularizer=l2(reg_lambda), activation='relu'))
        
        #encode each of the two inputs into a vector
        encoded_l = nnet(left_input)
        encoded_r = nnet(right_input)
        
        #merge two encoded inputs with the l1 distance between them
        distance = lambda x: K.abs(x[0] - x[1])
        merge = Lambda(distance)([encoded_l, encoded_r])
        merge = Dense(1, activation='sigmoid')(merge)
        siamese_net = KerasModel(input=[left_input, right_input], output=merge)
        siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = siamese_net
        self.name = 'siameseNN'
        
    def train(self, train_x, train_y, val_x, val_y, epochs=200, batch_size=1000, shuffle=True):

        early_stopping = EarlyStopping(monitor='val_acc', mode='auto', 
                           verbose=1, patience=10, min_delta=0, 
                           restore_best_weights=True)
        
        self.model.fit(train_x, train_y, 
                        validation_data=(val_x, val_y),
                        epochs=epochs, batch_size=batch_size, 
                        shuffle=shuffle, callbacks=[early_stopping])
        

    def save(self, file_name):
        
        self.model.save('weights/' + file_name + '.h5')

    def load(self, file_name):
        
        self.model = load_model('weights/' + file_name + '.h5')
        
    def predict(self, x):
        
        return self.model.predict(x)

    def score_non_parametric(self, test_samples, test_labels, support_set):
        
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
                    d += self.model.predict([samples[j], test_samples[i]])
                distance = d / len(idx)
                outputs.append(distance)
                labels.append(s)
            pred_idx = np.argmin(outputs)
            if test_labels[i] == labels[pred_idx]:
                correct += 1
            total += 1
           
        score = (correct/total)*100
        
        return score
  
    def score(self, x, y):
        
        return None
    
    