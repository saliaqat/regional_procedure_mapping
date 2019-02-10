from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from Models.model import Model

class BinaryLogisticRegressionModel(Model):
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
        fpr, tpr, thresholds = roc_curve(y, pred, pos_labels)
        return auc(fpr, tpr)


    def F1(self, x, y):
        pred = self.predict(x)
        return f1_score(y, pred, average='macro')
        pass



class MultiClassLogisticRegression(Model):
    def __init__(self, penalty='l2'):
        self.model = LogisticRegression(penalty=penalty, solver='newton-cg')
        self.name = 'multiclassLogisticRegressionModel'

    def train(self, x, y):
        self.model.fit(x, y, )

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def AUC(self, x, y, n_classes, pos_labels=None):
        raise NotImplementedError
        # roc = {label: [] for label in test_y['ON WG IDENTIFIER'].unique()}
        #
        # y_score = self.model.decision_function(x)
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # return roc_auc


    def plot_ROC(self, x, y, n_classes, pos_labels=None, output_dir='output/'):
        raise NotImplementedError
        # y_score = self.model.decision_function(x)
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # for i in range(n_classes):
        #     plt.figure()
        #     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        #     plt.plot([0, 1], [0, 1], 'k--')
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.05])
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.title('Receiver operating characteristic example')
        #     plt.legend(loc="lower right")
        #     plt.savefig(output_dir)


    def F1(self, x, y):
        pred = self.predict(x)
        return f1_score(y, pred, average='micro')

