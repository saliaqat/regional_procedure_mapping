from Models.logistic_regression import BinaryLogisticRegressionModel
from Models.model import Model
from data_reader import DataReader
from data_manipulator import *

def main():
    # Sample code to read and split data
    # data_reader = DataReader()
    # df = data_reader.get_all_data()
    # train_x, train_y, test_x, test_y= get_train_test_split(df)

    # Sample code to train logistic regression model
    # lg = BinaryLogisticRegressionModel()
    # lg.train(train_x, train_y)
    # evaluate_model(lg,test_x ,test_y , plot_roc=True)

    # Opens interactive terminal. Used to test out code.
    # from IPython import embed
    # embed()

    pass


def get_sample_data():
    train_x = [[1], [2], [3], [4], [5]]
    train_y = [1, 0, 1, 0, 1]
    test_x = [[3], [5], [6], [1], [2]]
    test_y = [1, 1, 0, 1, 0]
    return train_x, train_y, test_x, test_y

def evaluate_model(model, test_x, test_y, plot_roc=False):
    predictions = model.predict(test_x)
    score = model.score(test_x, test_y)
    AUC = model.AUC(test_x, test_y)
    F1 = model.F1(test_x, test_y)

    if plot_roc:
        model.plot_ROC(test_x, test_y)

    print(model.get_name() + ' Evaluation')
    print("predictions: ", predictions)
    print("score: ", score)
    print("auc: ", AUC)
    print("f1: ", F1)

if __name__ == '__main__':
    main()