import numpy as np
from Models.model import Model
from sklearn.neighbors import KNeighborsClassifier

class KNN(Model):
    def __init__(self, n_neighbors=10):

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.name = 'kNN'
        
    def train(self, x, y):

        self.model.fit(x, y)

    def predict(self, x):
        
        return self.model.predict(x)

    def score(self, x, y):
        
        return self.model.score(x, y)