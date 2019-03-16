from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from Models.model import Model

class RandomForest(Model):
    def __init__(self, penalty='l2', name='RandomForestModel'):
        self.model = RandomForestClassifier()
        self.name = name

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    # def AUC(self, x, y, pos_labels=None):
    #     pred = self.predict(x)
    #     fpr, tpr, thresholds = roc_curve(y, pred, pos_labels)
    #     return auc(fpr, tpr)

    def F1(self, x, y):
        pred = self.predict(x)
        return f1_score(y, pred, average='macro')
        pass