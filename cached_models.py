from cache_em_all import Cachable
import numpy as np
from Models.logistic_regression import BinaryLogisticRegressionModel, MultiClassLogisticRegression
from Models.random_forest import RandomForest
from Models.neural_net import MultiClassSimpleCNN
from Models.neural_net import MultiClassSimpleNN
from Models.model import Model
from Models.svm import SVM


@Cachable("get_multiclass_logistic_regression_model_bag_of_words_simple.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_simple(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_simple')
    lg.train(train_x, train_y)

    return lg

@Cachable("get_multiclass_logistic_regression_model_doc2vec_simple.pkl", version=1)
def _get_multiclass_logistic_regression_model_doc2vec_simple(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_doc2vec_simple')
    lg.train(train_x, train_y)

    return lg

@Cachable("get_multiclass_logistic_regression_model_doc2vec_simple_16384.pkl", version=1)
def _get_multiclass_logistic_regression_model_doc2vec_simple_16384(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_doc2vec_simple_16384')
    lg.train(train_x, train_y)

    return lg

@Cachable("get_svm_model_bag_of_words_simple.pkl", version=1)
def _get_svm_model_bag_of_words_simple(train_x, train_y):
    svm = SVM(name='SVM_bag_of_words_simple')
    svm.train(train_x, train_y)

    return svm

@Cachable("get_svm_model_doc2vec_simple.pkl", version=1)
def _get_svm_model_doc2vec_simple(train_x, train_y):
    svm = SVM(name='SVM_doc2vec_simple')
    svm.train(train_x, train_y)

    return svm

@Cachable("get_svm_model_doc2vec_simple_16384.pkl", version=1)
def _get_svm_model_doc2vec_simple_16384(train_x, train_y):
    svm = SVM(name='SVM_doc2vec_simple_16384')
    svm.train(train_x, train_y)

    return svm

@Cachable("get_random_forest_model_bag_of_words_simple.pkl", version=1)
def _get_random_forest_model_bag_of_words_simple(train_x, train_y):
    random_forest = RandomForest(name='random_forest_bag_of_words_simple')
    random_forest.train(train_x, train_y)

    return random_forest

@Cachable("get_random_forest_model_doc2vec_simple.pkl", version=1)
def _get_random_forest_model_doc2vec_simple(train_x, train_y):
    random_forest = RandomForest(name='random_forest_doc2vec_simple')
    random_forest.train(train_x, train_y)

    return random_forest

@Cachable("get_random_forest_model_doc2vec_simple_16384.pkl", version=1)
def _get_random_forest_model_doc2vec_simple_16384(train_x, train_y):
    random_forest = RandomForest(name='random_forest_doc2vec_simple_16384')
    random_forest.train(train_x, train_y)

    return random_forest

@Cachable("get_nn_model_bag_of_words_simple.pa", version=1)
def _get_nn_model_bag_of_words_simple(train_x, train_y, labels):
    model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model

@Cachable("get_nn_model_doc2vec_simple.pa", version=1)
def _get_nn_model_doc2vec_simple(train_x, train_y, labels):
    model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model

@Cachable("get_nn_model_doc2vec_simple_16384.pa", version=1)
def _get_nn_model_doc2vec_simple_16384(train_x, train_y, labels):
    model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model


#
#
# # 2048 input size
# @Cachable("multiclass_logistic_regression_model_word2doc.pkl", version=3)
# def _get_multiclass_logistic_regression_model_word2doc(train_x, train_y):
#     lg = MultiClassLogisticRegression()
#     lg.train(train_x, train_y)
#     return lg
#
# # Cacheable saves the model, so we don't have to train it again. (training takes around 15 minutes)
# # version 1 uses the default regex tokenizer and follows the most basic formulation
# # version 2 uses the default r'[\S]+' tokenizer and follows the most basic formulation
# @Cachable("multiclass_logistic_regression_model.pkl", version=1)
# def _get_multiclass_logistic_regression_model(train_x, train_y):
#     lg = MultiClassLogisticRegression()
#     lg.train(train_x, train_y)
#     return lg
#
#
# def _get_multiclass_random_forest_model(train_x, train_y):
#     lg = RandomForest()
#     lg.train(train_x, train_y)
#     return lg
#
#
# # @Cachable("multiclass_simple_cnn_model.pa", version=2)
# def _get_multiclass_simple_cnn_model(train_x, train_y, labels):
#     model = MultiClassSimpleCNN(train_x.shape, np.array(labels))
#     model.set_train_data(train_x, train_y)
#     model.train()
#     return model
#
# def _get_multiclass_simple_nn_model(train_x, train_y, labels):
#     model = MultiClassSimpleNN(train_x.shape, np.array(labels))
#     model.set_train_data(train_x, train_y)
#     model.train()
#     return model