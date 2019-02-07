import abc
from sklearn import metrics
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        pass

    def plot_ROC(self, x, y, pos_labels=None, output_dir='output/'):
        pred = self.predict(x)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_labels)
        auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b',
                 label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if output_dir is None:
            plt.show()
        else:
            plt.savefig(output_dir+self.name+'_ROC')

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def train(self, x, y):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, x, y):
        raise NotImplementedError

    @abc.abstractmethod
    def AUC(self):
        raise NotImplementedError

    @abc.abstractmethod
    def F1(self):
        raise NotImplementedError

