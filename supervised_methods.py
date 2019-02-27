import numpy as np
import pandas as pd
from Models.logistic_regression import BinaryLogisticRegressionModel, MultiClassLogisticRegression
from Models.random_forest import RandomForest
from Models.neural_net import MultiClassSimpleCNN
from Models.neural_net import MultiClassSimpleNN
from Models.model import Model
from data_reader import DataReader
from data_manipulator import *

from cache_em_all import Cachable

import warnings
warnings.filterwarnings("ignore")

# What Salaar is working on.

def main():
    supervised_scratch()

def supervised_scratch():
    # multiclass_logistic_regression()
    # random_forest()
    # convolutional_neural_network()
    neural_network()
    pass

# Baseline
# Score:    0.7962874821513565
# F1:       0.7962874821513564
def multiclass_logistic_regression():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)

    model = _get_multiclass_logistic_regression_model(train_x, train_y)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
    test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)

    evaluate_model(model, test_x, test_y, plot_roc=False)

#Random Forest
# Score:    0.6605697966954511
def random_forest():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)

    model = _get_multiclass_random_forest_model(train_x, train_y)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
    test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)

    evaluate_model(model, test_x, test_y, plot_roc=False)

# CNN
# Score:    0.011287142177194533
def convolutional_neural_network():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)

    model = _get_multiclass_simple_cnn_model(train_x, train_y, data_reader.get_region_labels()['Code'])

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
    test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

# NN
def neural_network():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_features(tokens, train_y_raw)

    model = _get_multiclass_simple_nn_model(train_x, train_y, data_reader.get_region_labels()['Code'])

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=True)
    test_x, test_y, _ = tokens_to_features(tokens, test_y_raw, feature_names=feature_names)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)



# Cacheable saves the model, so we don't have to train it again. (training takes around 15 minutes)
# version 1 uses the default regex tokenizer and follows the most basic formulation
# version 2 uses the default r'[\S]+' tokenizer and follows the most basic formulation
@Cachable("multiclass_logistic_regression_model.pkl", version=1)
def _get_multiclass_logistic_regression_model(train_x, train_y):
    lg = MultiClassLogisticRegression()
    lg.train(train_x, train_y)
    return lg


def _get_multiclass_random_forest_model(train_x, train_y):
    lg = RandomForest()
    lg.train(train_x, train_y)
    return lg


# @Cachable("multiclass_simple_cnn_model.pa", version=2)
def _get_multiclass_simple_cnn_model(train_x, train_y, labels):
    model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model

def _get_multiclass_simple_nn_model(train_x, train_y, labels):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model


def evaluate_model(model, test_x, test_y, plot_roc=False):
    predictions = model.predict(test_x)
    score = model.score(test_x, test_y)
#    AUC = model.AUC(test_x, test_y)
#     F1 = model.F1()

    # if plot_roc:
    #     model.plot_ROC(test_x, test_y)

    print(model.get_name() + ' Evaluation')
    print("predictions: ", predictions)
    print("score: ", score)
    # print("auc: ", AUC)
    # print("f1: ", F1)

def evaluate_model_nn(model, test_x, test_y, plot_roc=False):
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


def manual_testing(model, feature_names, data_reader):
    stdin = ''
    while stdin != 'quit':
        risproccode = np.NAN
        # risproccode = np.NaN if risproccode == '' else risproccode
        risprocdesc = input("Enter ris_proc_desc: ")
        risprocdesc = np.NaN if risprocdesc == '' else risprocdesc
        pacsproccode = np.NAN
        # pacsproccode = np.NaN if pacsproccode == '' else pacsproccode
        pacsprocdesc = input("Enter pacs_proc_desc: ")
        pacsprocdesc = np.NaN if pacsprocdesc == '' else pacsprocdesc
        pacsstuddesc = input("Enter pacs_study_desc: ")
        pacsstuddesc = np.NaN if pacsstuddesc == '' else pacsstuddesc
        pacsbodypart = input("Enter pacs_body_part: ")
        pacsbodypart = np.NaN if pacsbodypart == '' else risprocdesc
        pacsmodality = input("Enter pacs_modality: ")
        pacsmodality = np.NaN if pacsmodality == '' else pacsmodality

        from IPython import embed
        embed()

        text_columns = ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION', 'PACS SITE PROCEDURE CODE',
                        'PACS PROCEDURE DESCRIPTION',
                        'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY']
        df = pd.DataFrame([[risproccode, risprocdesc, pacsproccode, pacsprocdesc, pacsstuddesc, pacsbodypart,
                            pacsmodality]],
                          columns=text_columns)
        tokens, _ = tokenize_columns(df, 1, save_missing_feature_as_string=False)
        test_x, _, _ = tokens_to_features(tokens, 1, feature_names=feature_names)

        answer = int(model.predict(test_x))
        df = data_reader.get_all_data()
        df = df[df['ON WG IDENTIFIER'] == answer]
        print(answer)
        print(df)

# MANUAL TESTING FUNCTION
def manual_testing_nn(model, feature_names, data_reader):
    stdin = ''
    while stdin != 'quit':
        risproccode = np.NAN
        # risproccode = np.NaN if risproccode == '' else risproccode
        risprocdesc = input("Enter ris_proc_desc: ")
        risprocdesc = np.NaN if risprocdesc == '' else risprocdesc
        pacsproccode = np.NAN
        # pacsproccode = np.NaN if pacsproccode == '' else pacsproccode
        pacsprocdesc = input("Enter pacs_proc_desc: ")
        pacsprocdesc = np.NaN if pacsprocdesc == '' else pacsprocdesc
        pacsstuddesc = input("Enter pacs_study_desc: ")
        pacsstuddesc = np.NaN if pacsstuddesc == '' else pacsstuddesc
        pacsbodypart = input("Enter pacs_body_part: ")
        pacsbodypart = np.NaN if pacsbodypart == '' else risprocdesc
        pacsmodality = input("Enter pacs_modality: ")
        pacsmodality = np.NaN if pacsmodality == '' else pacsmodality

        # from IPython import embed
        # embed()

        text_columns = ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION', 'PACS SITE PROCEDURE CODE',
                        'PACS PROCEDURE DESCRIPTION',
                        'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY']
        df = pd.DataFrame([[risproccode, risprocdesc, pacsproccode, pacsprocdesc, pacsstuddesc, pacsbodypart,
                            pacsmodality]],
                          columns=text_columns)
        tokens, _ = tokenize_columns(df, 1, save_missing_feature_as_string=False)
        test_x, _, _ = tokens_to_features(tokens, 1, feature_names=feature_names)
        test_x = test_x.A
        test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

        prediction = model.predict(test_x)
        one_hot_answer = np.zeros_like(prediction)
        one_hot_answer[np.arange(len(prediction)), prediction.argmax(1)] = 1
        answer = int(model.encoder.inverse_transform(one_hot_answer))
        df = data_reader.get_all_data()
        df = df[df['ON WG IDENTIFIER'] == answer]
        print(answer)
        print(df)


if __name__ == '__main__':
    main()