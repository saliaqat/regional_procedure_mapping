from cache_em_all import Cachable
import numpy as np
from Models.logistic_regression import BinaryLogisticRegressionModel, MultiClassLogisticRegression
from Models.random_forest import RandomForest
from Models.neural_net import MultiClassSimpleCNN
from Models.neural_net import MultiClassSimpleNN, MultiClassNN, MultiClassNNScratch, MultiClassNNScratchAuto
from Models.naive_bayes import NaiveBayes, MultinomialNaiveBayes
from Models.model import Model
from Models.svm import SVM

# 0.7232610321615557
@Cachable("get_multiclass_logistic_regression_model_bag_of_words_simple.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_simple(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_simple')
    lg.train(train_x, train_y)

    return lg

# 0.7189093628884204
@Cachable("get_multiclass_logistic_regression_model_bag_of_words_full_no_repeat_no_short.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_full_no_repeat_no_short(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_full_no_repeat_no_short')
    lg.train(train_x, train_y)

    return lg

# 0.7189093628884204
@Cachable("get_multiclass_logistic_regression_model_bag_of_words_full_no_empty.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_full_no_empty(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_full_no_empty')
    lg.train(train_x, train_y)

    return lg

# 0.08125382470932209
@Cachable("get_multiclass_logistic_regression_model_doc2vec_simple.pkl", version=1)
def _get_multiclass_logistic_regression_model_doc2vec_simple(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_doc2vec_simple')
    lg.train(train_x, train_y)

    return lg

# too big, doc2vec not working well
@Cachable("get_multiclass_logistic_regression_model_doc2vec_simple_16384.pkl", version=1)
def _get_multiclass_logistic_regression_model_doc2vec_simple_16384(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_doc2vec_simple_16384')
    lg.train(train_x, train_y)

    return lg

#0.797715373631604
@Cachable("get_multiclass_logistic_regression_model_bag_of_words_full.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_full(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_full')
    lg.train(train_x, train_y)

    return lg

@Cachable("get_multiclass_logistic_regression_model_tfidf.pkl", version=1)
def _get_multiclass_logistic_regression_model_tfidf(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_tfidf')
    lg.train(train_x, train_y)

    return lg

# 0.7962874821513565
@Cachable("get_multiclass_logistic_regression_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_multiclass_logistic_regression_model_bag_of_words_full_save_missing(train_x, train_y):
    lg = MultiClassLogisticRegression(name='multiclass_logistic_regression_bag_of_words_full_save_missing')
    lg.train(train_x, train_y)

    return lg

# SVM underperforming
@Cachable("get_svm_model_bag_of_words_simple.pkl", version=1)
def _get_svm_model_bag_of_words_simple(train_x, train_y):
    svm = SVM(name='SVM_bag_of_words_simple')
    svm.train(train_x, train_y)

    return svm

# SVM underperforming
@Cachable("get_svm_model_doc2vec_simple.pkl", version=1)
def _get_svm_model_doc2vec_simple(train_x, train_y):
    svm = SVM(name='SVM_doc2vec_simple')
    svm.train(train_x, train_y)

    return svm

# SVM underperforming
@Cachable("get_svm_model_doc2vec_simple_16384.pkl", version=1)
def _get_svm_model_doc2vec_simple_16384(train_x, train_y):
    svm = SVM(name='SVM_doc2vec_simple_16384')
    svm.train(train_x, train_y)

    return svm

# SVM underperforming
@Cachable("get_svm_model_bag_of_words_full.pkl", version=1)
def _get_svm_model_bag_of_words_full(train_x, train_y):
    svm = SVM(name='SVM_bag_of_words_full')
    svm.train(train_x, train_y)

    return svm

# SVM underperforming
@Cachable("get_svm_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_svm_model_bag_of_words_full_save_missing(train_x, train_y):
    svm = SVM(name='SVM_bag_of_words_full_save_missing')
    svm.train(train_x, train_y)

    return svm
# 0.649962602842184
@Cachable("get_random_forest_model_bag_of_words_simple.pkl", version=1)
def _get_random_forest_model_bag_of_words_simple(train_x, train_y):
    random_forest = RandomForest(name='random_forest_bag_of_words_simple')
    random_forest.train(train_x, train_y)

    return random_forest

# 0.07690215543618685
@Cachable("get_random_forest_model_doc2vec_simple.pkl", version=1)
def _get_random_forest_model_doc2vec_simple(train_x, train_y):
    random_forest = RandomForest(name='random_forest_doc2vec_simple')
    random_forest.train(train_x, train_y)

    return random_forest

# Didn't run, too big, doc2vec not performing well
@Cachable("get_random_forest_model_doc2vec_simple_16384.pkl", version=1)
def _get_random_forest_model_doc2vec_simple_16384(train_x, train_y):
    random_forest = RandomForest(name='random_forest_doc2vec_simple_16384')
    random_forest.train(train_x, train_y)

    return random_forest

#0.7329842931937173
@Cachable("get_random_forest_model_bag_of_words_full.pkl", version=1)
def _get_random_forest_model_bag_of_words_full(train_x, train_y):
    random_forest = RandomForest(name='random_forest_bag_of_words_full')
    random_forest.train(train_x, train_y)

    return random_forest

@Cachable("get_random_forest_model_tfidf.pkl", version=1)
def _get_random_forest_model_tfidf(train_x, train_y):
    random_forest = RandomForest(name='random_forest_tfidf')
    random_forest.train(train_x, train_y)

    return random_forest
# 0.7076902155436187
@Cachable("get_random_forest_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_random_forest_model_bag_of_words_full_save_missing(train_x, train_y):
    random_forest = RandomForest(name='random_forest_bag_of_words_full_save_missing')
    random_forest.train(train_x, train_y)

    return random_forest

#0.28095464744679405
# @Cachable("get_nn_model_bag_of_words_simple.pkl", version=1)
def _get_nn_model_bag_of_words_simple(train_x, train_y, labels, epochs=50):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels), epochs=epochs)
    model.set_train_data(train_x, train_y)
    model.train()
    return model

# @Cachable("get_nn_model_bag_of_words_simple_v2.pkl", version=1)
def _get_nn_model_bag_of_words_simple_v2(train_x, train_y, labels, epochs=50, batch_size=64):
    model = MultiClassNN(train_x.shape, np.array(labels), epochs=epochs, batch_size=batch_size)
    model.set_train_data(train_x, train_y)
    model.train()
    return model

# @Cachable("get_nn_model_bag_of_words_simple_v2.pkl", version=1)
def _get_nn_model_bag_of_words_simple_scratch(train_x, train_y, val_x, val_y, labels, epochs=50, batch_size=64):
    model = MultiClassNNScratch(train_x.shape, np.array(labels), epochs=epochs, batch_size=batch_size)
    model.set_train_data(train_x, train_y)
    model.train(val_x, val_y)
    return model

# @Cachable("get_nn_model_bag_of_words_simple_v2.pkl", version=1)
def _get_nn_model_bag_of_words_simple_scratch_auto(train_x, train_y, val_x, val_y, labels, epochs=50, batch_size=64):
    model = MultiClassNNScratchAuto(train_x.shape, np.array(labels), epochs=epochs, batch_size=batch_size)
    model.set_train_data(train_x, train_y)
    model.train(val_x, val_y)
    return model

# @Cachable("get_nn_model_bag_of_words_simple_v2.pkl", version=1)
def _get_nn_model_bag_of_words_simple_big(train_x, train_y, labels, epochs=50):
    model = MultiClassNN(train_x.shape, np.array(labels), epochs=epochs)
    model.set_train_data(train_x, train_y)
    model.train()
    return model

# 0.0114911265674372748
# @Cachable("get_nn_model_doc2vec_simple.pkl", version=1)
def _get_nn_model_doc2vec_simple(train_x, train_y, labels):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels))
    model.set_train_data_np(train_x, train_y)
    model.train()
    return model

# 0.012171074998300129
# @Cachable("get_nn_model_doc2vec_simple_16384.pkl", version=1)
def _get_nn_model_doc2vec_simple_16384(train_x, train_y, labels):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels))
    model.set_train_data_np(train_x, train_y)
    model.train()
    return model

# 0.24797715373631604
# @Cachable("get_nn_model_bag_of_words_full.pkl", version=1)
def _get_nn_model_bag_of_words_full(train_x, train_y, labels):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model

# 0.22880261100156388
# @Cachable("get_nn_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_nn_model_bag_of_words_full_save_missing(train_x, train_y, labels):
    model = MultiClassSimpleNN(train_x.shape, np.array(labels))
    model.set_train_data(train_x, train_y)
    model.train()
    return model

# 0.5317195893112123
@Cachable("get_naive_bayes_model_bag_of_words_simple.pkl", version=1)
def _get_naive_bayes_model_bag_of_words_simple(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_bag_of_words_simple')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

# Threw error
@Cachable("get_naive_bayes_model_doc2vec_simple.pkl", version=1)
def _get_naive_bayes_model_doc2vec_simple(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_doc2vec_simple')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

# Threw error
@Cachable("get_naive_bayes_model_doc2vec_simple_16384.pkl", version=1)
def _get_naive_bayes_model_doc2vec_simple_16384(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_doc2vec_simple_16384')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

# 0.7297205412388659
@Cachable("get_naive_bayes_model_bag_of_words_full.pkl", version=1)
def _get_naive_bayes_model_bag_of_words_full(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_bag_of_words_full')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

@Cachable("get_naive_bayes_model_tfidf.pkl", version=1)
def _get_naive_bayes_model_tfidf(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_tfidf')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

# 0.7296525464064731
@Cachable("get_naive_bayes_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_naive_bayes_model_bag_of_words_full_save_missing(train_x, train_y):
    naive_bayes = NaiveBayes(name='NaiveBayes_bag_of_words_full_save_missing')
    naive_bayes.train(train_x, train_y)

    return naive_bayes

# 0.504045692527368
@Cachable("get_multinomial_naive_bayes_model_bag_of_words_simple.pkl", version=1)
def _get_multinomial_naive_bayes_model_bag_of_words_simple(train_x, train_y):
    multinomial_naive_bayes = MultinomialNaiveBayes(name='MultinomialNaiveBayes_bag_of_words_simple')
    multinomial_naive_bayes.train(train_x, train_y)

    return multinomial_naive_bayes
# Threw error
@Cachable("get_multinomial_naive_bayes_model_doc2vec_simple.pkl", version=1)
def _get_multinomial_naive_bayes_model_doc2vec_simple(train_x, train_y):
    multinomial_naive_bayes = MultinomialNaiveBayes(name='MultinomialNaiveBayes_doc2vec_simple')
    multinomial_naive_bayes.train(train_x, train_y)

    return multinomial_naive_bayes

# Threw error
@Cachable("get_multinomial_naive_bayes_model_doc2vec_simple_16384.pkl", version=1)
def _get_multinomial_naive_bayes_model_doc2vec_simple_16384(train_x, train_y):
    multinomial_naive_bayes = MultinomialNaiveBayes(name='MultinomialNaiveBayes_doc2vec_simple_16384')
    multinomial_naive_bayes.train(train_x, train_y)

    return multinomial_naive_bayes

# 0.5563337186373836
@Cachable("get_multinomial_naive_bayes_model_bag_of_words_full.pkl", version=1)
def _get_multinomial_naive_bayes_model_bag_of_words_full(train_x, train_y):
    multinomial_naive_bayes = MultinomialNaiveBayes(name='MultinomialNaiveBayes_bag_of_words_full')
    multinomial_naive_bayes.train(train_x, train_y)

    return multinomial_naive_bayes

# 0.4754878629224179
@Cachable("get_multinomial_naive_bayes_model_bag_of_words_full_save_missing.pkl", version=1)
def _get_multinomial_naive_bayes_model_bag_of_words_full_save_missing(train_x, train_y):
    multinomial_naive_bayes = MultinomialNaiveBayes(name='MultinomialNaiveBayes_bag_of_words_full_save_missing')
    multinomial_naive_bayes.train(train_x, train_y)

    return multinomial_naive_bayes
#
#
# # 2048 input size
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
# # @Cachable("multiclass_simple_cnn_model.pkl", version=2)
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