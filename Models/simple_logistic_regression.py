from sklearn.linear_model import LogisticRegression
from Models.model import Model
from sklearn import metrics
from matplotlib import pyplot as plt

class LogisticRegressionModel(Model):

    def __init__(self, penalty='l2'):
        self.model = LogisticRegression(penalty=penalty)
        self.name = 'binaryLogisticRegressionModel'

    def train(self, x, y):
        self.model.fit(x, y, )

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def AUC(self, x, y, pos_labels=None):
        pred = self.predict(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_labels)
        return metrics.auc(fpr, tpr)


    def F1(self, x, y):
        pred = self.predict(x)
        return metrics.f1_score(y, pred, average='macro')
        pass
