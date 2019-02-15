import numpy as np
from Models.logistic_regression import BinaryLogisticRegressionModel, MultiClassLogisticRegression
from Models.neural_net import MultiClassSimpleCNN
from Models.model import Model
from data_reader import DataReader
from data_manipulator import *

from cache_em_all import Cachable

# What Salaar is working on.

def main():
    supervised_scratch()

def supervised_scratch():
    multiclass_logistic_regression()
    pass

# Baseline
# Score:    0.7962874821513565
# F1:       0.7962874821513564
def multiclass_logistic_regression():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=True)
    train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)

    # model = _get_multiclass_logistic_regression_model(train_x, train_y)

    # from IPython import embed
    # embed()

    model = _get_multiclass_simple_cnn_model(train_x, train_y, data_reader.get_region_labels()['Code'])

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
    test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)

    evaluate_model(model, test_x, test_y, plot_roc=False)

# Cacheable saves the model, so we don't have to train it again. (training takes around 15 minutes)
@Cachable("multiclass_logistic_regression_model.pkl", version=1)
def _get_multiclass_logistic_regression_model(train_x, train_y):
    lg = MultiClassLogisticRegression()
    lg.train(train_x, train_y)
    return lg

# @Cachable("multiclass_simple_cnn_model.pkl", version=2)
def _get_multiclass_simple_cnn_model(train_x, train_y, labels):
    model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model


def evaluate_model(model, test_x, test_y, plot_roc=False):
    model.set_test_data(test_x, test_y)
    predictions = model.predict()
    score = model.score()
#    AUC = model.AUC(test_x, test_y)
#     F1 = model.F1()

    # if plot_roc:
    #     model.plot_ROC(test_x, test_y)

    print(model.get_name() + ' Evaluation')
    print("predictions: ", predictions)
    print("score: ", score)
    # print("auc: ", AUC)
    # print("f1: ", F1)

if __name__ == '__main__':
    main()