from Models.simple_logistic_regression import LogisticRegressionModel
from Models.model import Model
from data_reader import DataReader

def main():
    lg = LogisticRegressionModel()
    lg.train([[1], [2], [3], [4], [5]], [1, 0, 1, 0, 1])
    evaluate_model(lg, [[3], [5], [6], [1], [2]], [1, 1, 0, 1, 0])
    pass

def evaluate_model(model, test_x, test_y, plot_roc=False):
    predictions = model.predict(test_x)
    score = model.score(test_x, test_y)
    AUC = model.AUC(test_x, test_y)
    if plot_roc:
        model.plot_ROC(test_x, test_y)
    F1 = model.F1(test_x, test_y)
    print(model.get_name() + ' Evaluation')
    print("predictions: ", predictions)
    print("score: ", score)
    print("auc: ", AUC)
    print("f1: ", F1)

if __name__ == '__main__':
    main()