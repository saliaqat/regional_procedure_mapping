from Models.model import Model
from sklearn.svm import SVC


class SVM(Model):
    def __init__(self, penalty='l2', name='SVM'):
        self.model = SVC()
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

