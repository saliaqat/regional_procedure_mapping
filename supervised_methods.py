import numpy as np
import pandas as pd

from data_reader import DataReader
from data_manipulator import *
from data_manipulator_interface import *
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_full_no_repeat_no_short
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_simple
from cached_models import _get_multiclass_logistic_regression_model_doc2vec_simple
from cached_models import _get_multiclass_logistic_regression_model_doc2vec_simple_16384
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_full
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_full_save_missing
from cached_models import _get_svm_model_bag_of_words_simple
from cached_models import _get_svm_model_doc2vec_simple
from cached_models import _get_svm_model_doc2vec_simple_16384
from cached_models import _get_svm_model_bag_of_words_full
from cached_models import _get_svm_model_bag_of_words_full_save_missing
from cached_models import _get_random_forest_model_bag_of_words_simple
from cached_models import _get_random_forest_model_doc2vec_simple
from cached_models import _get_random_forest_model_doc2vec_simple_16384
from cached_models import _get_random_forest_model_bag_of_words_full
from cached_models import _get_random_forest_model_bag_of_words_full_save_missing
from cached_models import _get_nn_model_bag_of_words_simple
from cached_models import _get_nn_model_bag_of_words_simple_v2
from cached_models import _get_nn_model_bag_of_words_simple_big
from cached_models import _get_nn_model_doc2vec_simple
from cached_models import _get_nn_model_doc2vec_simple_16384
from cached_models import _get_nn_model_bag_of_words_full
from cached_models import _get_nn_model_bag_of_words_full_save_missing
from cached_models import _get_naive_bayes_model_bag_of_words_simple
from cached_models import _get_naive_bayes_model_doc2vec_simple
from cached_models import _get_naive_bayes_model_doc2vec_simple_16384
from cached_models import _get_naive_bayes_model_bag_of_words_full
from cached_models import _get_naive_bayes_model_bag_of_words_full_save_missing
from cached_models import _get_multinomial_naive_bayes_model_bag_of_words_simple
from cached_models import _get_multinomial_naive_bayes_model_doc2vec_simple
from cached_models import _get_multinomial_naive_bayes_model_doc2vec_simple_16384
from cached_models import _get_multinomial_naive_bayes_model_bag_of_words_full
from cached_models import _get_multinomial_naive_bayes_model_bag_of_words_full_save_missing

import warnings
warnings.filterwarnings("ignore")

# What Salaar is working on.

def main():
    supervised_scratch()
    # load_data()

# 80 percent with bag of words, full on _get_nn_model_bag_of_words_simple_v2 (single layer dense)
# 80 percent with bag of words full on _get_nn_model_bag_of_words_simple (multi layer dense)
def supervised_scratch():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    unlabelled_df = data_reader.get_east_dir()
    unlabelled_df = normalize_east_dir_df(unlabelled_df)

    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model  = _get_nn_model_bag_of_words_simple_v2(train_x, train_y, data_reader.get_region_labels()['Code'],
                                                      epochs=30, batch_size=128)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

    from IPython import embed
    embed()
    tokens_train, _ = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    _, _, feature_names = tokens_to_bagofwords(tokens_train, _)

    tokens, _ = tokenize(unlabelled_df, _, save_missing_feature_as_string=False)
    semi_x_base, _, _ = tokens_to_bagofwords(tokens, _, feature_names=feature_names)
    train_threshold = 0.99

    for i in range(10):
        m = model.model
        pred = m.predict(semi_x_base)
        semi_y = np.zeros_like(pred)
        semi_y[np.arange(len(pred)), pred.argmax(1)] = 1
        semi_y = semi_y[pred.max(axis=1) > train_threshold]
        semi_x = semi_x_base[pred.max(axis=1) > train_threshold]

        m.fit(semi_x, semi_y, batch_size=128, epochs=50)

        evaluate_model_nn(model, test_x, test_y, plot_roc=False)
        semi_x_base = semi_x_base[~(pred.max(axis=1) > train_threshold)]

    from IPython import embed
    embed()

def load_data():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_save_missing_features_remove_repeats_remove_short(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_simple_4096(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_save_missing_features_remove_repeats_remove_short_4096(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_simple_8192(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_save_missing_features_remove_repeats_remove_short_8192(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    doc2vec_save_missing_features_remove_repeats_remove_short_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

def run_all_models():
    # print('multiclass_logistic_regression_simple_bag_of_words')
    # multiclass_logistic_regression_simple_bag_of_words()
    # print('multiclass_random_forest_simple_bag_of_words')
    # multiclass_random_forest_simple_bag_of_words()
    # print('multiclass_naive_bayes_simple_bag_of_words')
    # multiclass_naive_bayes_simple_bag_of_words()
    # print('multiclass_multinomial_naive_bayes_simple_bag_of_words')
    # multiclass_multinomial_naive_bayes_simple_bag_of_words()
    # print('multiclass_nn_simple_bag_of_words')
    # multiclass_nn_simple_bag_of_words()
    # print('multiclass_svm_simple_bag_of_words')
    # multiclass_svm_simple_bag_of_words()

    # print('multiclass_logistic_regression_full_bag_of_words')
    # multiclass_logistic_regression_full_bag_of_words()
    # print('multiclass_random_forest_full_bag_of_words')
    # multiclass_random_forest_full_bag_of_words()
    # print('multiclass_naive_bayes_full_bag_of_words')
    # multiclass_naive_bayes_full_bag_of_words()
    # print('multiclass_multinomial_naive_bayes_full_bag_of_words')
    # multiclass_multinomial_naive_bayes_full_bag_of_words()
    # print('multiclass_nn_full_bag_of_words')
    # multiclass_nn_full_bag_of_words()
    # print('multiclass_svm_full_bag_of_words')
    # multiclass_svm_full_bag_of_words()

    # print('multiclass_logistic_regression_full_save_missing_bag_of_words')
    # multiclass_logistic_regression_full_save_missing_bag_of_words()
    # print('multiclass_random_forest_full_save_missing_bag_of_words')
    # multiclass_random_forest_full_save_missing_bag_of_words()
    # print('multiclass_naive_bayes_full_save_missing_bag_of_words')
    # multiclass_naive_bayes_full_save_missing_bag_of_words()
    # print('multiclass_multinomial_naive_bayes_full_save_missing_bag_of_words')
    # multiclass_multinomial_naive_bayes_full_save_missing_bag_of_words()
    # print('multiclass_nn_full_save_missing_bag_of_words')
    # multiclass_nn_full_save_missing_bag_of_words()
    # print('multiclass_svm_full_save_missing_bag_of_words')
    # multiclass_svm_full_save_missing_bag_of_words()

    # print('multiclass_logistic_regression_doc2vec')
    # multiclass_logistic_regression_doc2vec()
    # print('multiclass_random_forest_doc2vec')
    # multiclass_random_forest_doc2vec()
    # print('multiclass_naive_bayes_doc2vec')
    # multiclass_naive_bayes_doc2vec()
    # print('multiclass_multinomial_naive_bayes_doc2vec')
    # multiclass_multinomial_naive_bayes_doc2vec()
    # print('multiclass_nn_doc2vec')
    # multiclass_nn_doc2vec()
    # print('multiclass_svm_doc2vec')
    # multiclass_svm_doc2vec()

    # multiclass_logistic_regression_doc2vec_16384()
    # multiclass_random_forest_doc2vec_16384()
    # multiclass_nn_doc2vec_16384()
    # multiclass_svm_doc2vec_16384()

    # multiclass_logistic_regression_full_bag_of_words_no_repeats_no_short()
    pass



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

def multiclass_logistic_regression_full_bag_of_words_no_repeats_no_short():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_bag_of_words_full_no_repeat_no_short(train_x, train_y)

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

def multiclass_logistic_regression_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_bag_of_words_full(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_logistic_regression_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multiclass_logistic_regression_model_bag_of_words_full_save_missing(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)
####################################################
# SVM
####################################################
#0.027673896783844427
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

def multiclass_svm_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_svm_model_bag_of_words_full(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_svm_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_svm_model_bag_of_words_full_save_missing(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

####################################################
# Random Forest
####################################################
#0.6850539878969198
def multiclass_random_forest_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)
#0.07799007275447066
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

def multiclass_random_forest_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_bag_of_words_full(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_random_forest_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_random_forest_model_bag_of_words_full_save_missing(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

####################################################
# Neural Networks
####################################################
#0.011627116339158224
def multiclass_nn_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_bag_of_words_simple(train_x, train_y, data_reader.get_region_labels()['Code'])

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)
#0.008499354049092269
def multiclass_nn_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_doc2vec_simple(train_x, train_y, data_reader.get_region_labels()['Code'])

    evaluate_model_nn_np(model, test_x, test_y, plot_roc=False)

def multiclass_nn_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_doc2vec_simple_16384(train_x, train_y, data_reader.get_region_labels()['Code'])

    evaluate_model_nn_np(model, test_x, test_y, plot_roc=False)

def multiclass_nn_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_bag_of_words_full(train_x, train_y, data_reader.get_region_labels()['Code'])

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

def multiclass_nn_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_nn_model_bag_of_words_full_save_missing(train_x, train_y, data_reader.get_region_labels()['Code'])

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

#####################
# NAIVE BAYES
#####################
def multiclass_naive_bayes_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    train_x, test_x = train_x.toarray(), test_x.toarray()

    model = _get_naive_bayes_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)
# No good. Naive bayes takes positive only. may want to transform, but doc2vec underperforming already
def multiclass_naive_bayes_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_naive_bayes_model_doc2vec_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

# No good. Naive bayes takes positive only. may want to transform, but doc2vec underperforming already
def multiclass_naive_bayes_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_naive_bayes_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_naive_bayes_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    train_x, test_x = train_x.toarray(), test_x.toarray()

    model = _get_naive_bayes_model_bag_of_words_full(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_naive_bayes_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    train_x, test_x = train_x.toarray(), test_x.toarray()

    model = _get_naive_bayes_model_bag_of_words_full_save_missing(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

#####################
# MULTINOMIAL NAIVE BAYES
#####################
def multiclass_multinomial_naive_bayes_simple_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multinomial_naive_bayes_model_bag_of_words_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

# No good. Naive bayes takes positive only. may want to transform, but doc2vec underperforming already
def multiclass_multinomial_naive_bayes_doc2vec():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multinomial_naive_bayes_model_doc2vec_simple(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

# No good. Naive bayes takes positive only. may want to transform, but doc2vec underperforming already
def multiclass_multinomial_naive_bayes_doc2vec_16384():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multinomial_naive_bayes_model_doc2vec_simple_16384(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_multinomial_naive_bayes_full_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multinomial_naive_bayes_model_bag_of_words_full(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)

def multiclass_multinomial_naive_bayes_full_save_missing_bag_of_words():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_save_missing(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model = _get_multinomial_naive_bayes_model_bag_of_words_full_save_missing(train_x, train_y)

    evaluate_model(model, test_x, test_y, plot_roc=False)


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

def evaluate_model_nn_np(model, test_x, test_y, plot_roc=False):
    model.set_test_data_np(test_x, test_y)
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