from sklearn.linear_model import LogisticRegression
from Models.model import Model
from sklearn import metrics
from matplotlib import pyplot as plt

class LogisticRegressionModel(Model):

    def __init__(self, penalty='l2', solver='newton-cg'):
        self.model = LogisticRegression(penalty=penalty, solver=solver)
        self.name = 'logisticRegressionModel'

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def AUC(self, x, y, pos_labels):
        pred = self.predict(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_labels)
        return metrics.auc(fpr, tpr)


    def F1(self):
        pass
