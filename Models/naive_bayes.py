from Models.model import Model
from sklearn.naive_bayes import GaussianNB, MultinomialNB

class NaiveBayes(Model):
    def __init__(self, name='GaussianNB'):
        self.model = GaussianNB()
        self.name = name

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def AUC(self, x, y, n_classes, pos_labels=None):
        raise NotImplementedError


    def plot_ROC(self, x, y, n_classes, pos_labels=None, output_dir='output/'):
        raise NotImplementedError

    def F1(self, x, y):
        pred = self.predict(x)
        return f1_score(y, pred, average='micro')

class MultinomialNaiveBayes(Model):
    def __init__(self, name='MultiNomialGaussianNB'):
        self.model = MultinomialNB()
        self.name = name

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def AUC(self, x, y, n_classes, pos_labels=None):
        raise NotImplementedError


    def plot_ROC(self, x, y, n_classes, pos_labels=None, output_dir='output/'):
        raise NotImplementedError

    def F1(self, x, y):
        pred = self.predict(x)
        return f1_score(y, pred, average='micro')

