import numpy as np
import pandas as pd

from data_reader import DataReader
from data_manipulator import *
from data_manipulator_interface import *
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_simple
from cached_models import _get_multiclass_logistic_regression_model_doc2vec_simple
from cached_models import _get_multiclass_logistic_regression_model_doc2vec_simple_16384
from cached_models import _get_svm_model_bag_of_words_simple
from cached_models import _get_svm_model_doc2vec_simple
from cached_models import _get_svm_model_doc2vec_simple_16384
from cached_models import _get_random_forest_model_bag_of_words_simple
from cached_models import _get_random_forest_model_doc2vec_simple
from cached_models import _get_random_forest_model_doc2vec_simple_16384
from cached_models import _get_nn_model_bag_of_words_simple
from cached_models import _get_nn_model_doc2vec_simple
from cached_models import _get_nn_model_doc2vec_simple_16384


import warnings
warnings.filterwarnings("ignore")

# What Salaar is working on.

def main():
    supervised_scratch()

def supervised_scratch():
    multiclass_logistic_regression_simple_bag_of_words()
    multiclass_logistic_regression_doc2vec()
    multiclass_logistic_regression_doc2vec_16384()

    multiclass_svm_simple_bag_of_words()
    multiclass_svm_doc2vec()
    multiclass_svm_doc2vec_16384()

def scratch():

    pass

####################################################
# LOGISTIC REGRESSION
####################################################
# 0.7232610321615557
def multiclass_logistic_regression_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

#0.09009315292037805
def multiclass_logistic_regression_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_doc2vec_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_logistic_regression_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

####################################################
# SVM
####################################################
def multiclass_svm_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_svm_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_svm_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_svm_model_doc2vec_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_svm_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_svm_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

####################################################
# Random Forest
####################################################
def multiclass_random_forest_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_random_forest_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_doc2vec_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_random_forest_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

####################################################
# Neural Networks
####################################################
def multiclass_nn_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_bag_of_words_simple(train_x, train_y)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

def multiclass_nn_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_doc2vec_simple(train_x, train_y)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

def multiclass_nn_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)





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
        test_x, _, _ = tokens_to_bagofwords(tokens, 1, feature_names=feature_names)

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
        test_x, _, _ = tokens_to_bagofwords(tokens, 1, feature_names=feature_names)
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