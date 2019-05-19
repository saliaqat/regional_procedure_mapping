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
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_full_no_empty
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
from cached_models import _get_nn_model_bag_of_words_simple_scratch
from cached_models import _get_nn_model_bag_of_words_simple_scratch_auto
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
from cached_models import _get_multiclass_logistic_regression_model_tfidf
from cached_models import _get_random_forest_model_tfidf
from cached_models import _get_naive_bayes_model_tfidf

from run_autoencoder import get_encoder
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random, string
import warnings
warnings.filterwarnings("ignore")

# What Salaar is working on.

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ktf.set_session(tf.Session(config=config))
    eval()

    # self_training_site_split()

    # neural_net_scratch_bag_of_words(bag_of_words_full_no_empty_val)
    # supervised_scratch()

def eval_ae():
    from Models.logistic_regression import MultiClassLogisticRegression
    from Models.random_forest import RandomForest
    from Models.naive_bayes import NaiveBayes
    from Models.svm import SVM

    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y = bag_of_words_full_no_empty_val_no_num_no_short_no_repeat(
        train_x_raw, train_y_raw,
        val_x_raw, val_y_raw, test_x_raw,
        test_y_raw)

    encoder = get_encoder(train_x, test_x, 4096)
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)
    encoded_val = encoder.predict(val_x)

    print('neural net ae')
    model = _get_nn_model_bag_of_words_simple_scratch(encoded_train, train_y, encoded_val, val_y,
                                                      data_reader.get_region_labels()['Code'], epochs=100,
                                                      batch_size=256)
    eval_nn(model, encoded_test, test_y)
    evaluate_model_nn(model, encoded_test, test_y)
    print('logistic regression ae')
    model = MultiClassLogisticRegression()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

    print('random forest ae')
    model = RandomForest()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

    print('naive bayes ae')
    model = NaiveBayes()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

def eval_pub_med():
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    tokens_train, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False,
                                       remove_empty=True)
    avg = []
    for item in tokens_train:
        words = []
        for word in item:
            if word in model.wv.vocab:
                vec = model.get_vector(word)
                words.append(vec)
        average = np.average(np.array(words), axis=0)
        if type(average) == np.float64:
            print('****')
            print(average)
            avg.append(np.zeros(200))
        else:
            avg.append(list(average))
    pub_med_train = np.array(avg)

    tokens_val, val_y_raw = tokenize(val_x_raw, val_y_raw, save_missing_feature_as_string=False, remove_empty=True)
    avg = []
    for item in tokens_val:
        words = []
        for word in item:
            if word in model.wv.vocab:
                vec = model.get_vector(word)
                words.append(vec)
        average = np.average(np.array(words), axis=0)
        if type(average) == np.float64:
            print('****')
            print(average)
            avg.append(np.zeros(200))
        else:
            avg.append(list(average))
    pub_med_val = np.array(avg)

    tokens_test, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False, remove_empty=True)
    avg = []
    for item in tokens_test:
        words = []
        for word in item:
            if word in model.wv.vocab:
                vec = model.get_vector(word)
                words.append(vec)
        average = np.average(np.array(words), axis=0)
        if type(average) == np.float64:
            print('****')
            print(average)
            avg.append(np.zeros(200))
        else:
            avg.append(list(average))
    pub_med_test = np.array(avg)

    print("pubmed, nn")
    nn_model = _get_nn_model_bag_of_words_simple_scratch(pub_med_train, train_y_raw, pub_med_val, val_y_raw,
                                                         data_reader.get_region_labels()['Code'], epochs=100, batch_size=256)
    eval_model(nn_model, pub_med_test, test_y_raw)
    print("pubmed, logistic regression")
    from Models.logistic_regression import MultiClassLogisticRegression
    log_reg = MultiClassLogisticRegression()
    log_reg.train(pub_med_train, train_y_raw)
    eval_model(log_reg, pub_med_test, test_y_raw)
    print("pubmed, random forest")
    from Models.random_forest import RandomForest
    rand_for = RandomForest()
    rand_for.train(pub_med_train, train_y_raw)
    eval_model(rand_for, pub_med_test, test_y_raw)
    print("pubmed, naivebayes")
    from Models.naive_bayes import NaiveBayes
    nb = NaiveBayes()
    nb.train(pub_med_train, train_y_raw)
    eval_model(nb, pub_med_test, test_y_raw)

def eval_nn():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw_v, train_y_raw_v, val_x_raw_v, val_y_raw_v, test_x_raw_v, test_y_raw_v = get_train_validate_test_split(df)
    train_x_v, train_y_v, val_x_v, val_y_v, test_x_v, test_y_v = bag_of_words_full_no_empty_val(train_x_raw_v, train_y_raw_v,
                                                                              val_x_raw_v, val_y_raw_v, test_x_raw_v, test_y_raw_v)
    nn_model = _get_nn_model_bag_of_words_simple_scratch(train_x_v, train_y_raw_v, val_x_v, val_y_v,
                                                         data_reader.get_region_labels()['Code'], epochs=100,
                                                         batch_size=256)
    print("bow, nn")
    eval_model(nn_model, test_x_v, test_y_v)
    train_x_v, train_y_v, val_x_v, val_y_v, test_x_v, test_y_v = tfidf_no_empty_val(train_x_raw_v,
                                                                                                train_y_raw_v,
                                                                                                val_x_raw_v,
                                                                                                val_y_raw_v,
                                                                                                test_x_raw_v,
                                                                                                test_y_raw_v)
    nn_model = _get_nn_model_bag_of_words_simple_scratch(train_x_v, train_y_raw_v, val_x_v, val_y_v,
                                                         data_reader.get_region_labels()['Code'], epochs=100,
                                                         batch_size=256)
    print("tfidf, nn")
    eval_model(nn_model, test_x_v, test_y_v)
    train_x_v, train_y_v, val_x_v, val_y_v, test_x_v, test_y_v = doc2vec_simple_val(train_x_raw_v,
                                                                                                train_y_raw_v,
                                                                                                val_x_raw_v,
                                                                                                val_y_raw_v,
                                                                                                test_x_raw_v,
                                                                                                test_y_raw_v)
    nn_model = _get_nn_model_bag_of_words_simple_scratch(train_x_v, train_y_raw_v, val_x_v, val_y_v,
                                                         data_reader.get_region_labels()['Code'], epochs=100,
                                                         batch_size=256)
    print("doc2vec, nn")
    eval_model(nn_model, test_x_v, test_y_v)

def eval():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)
    train_x, train_y, test_x, test_y = bag_of_words_full_no_empty(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    log_reg_bow = _get_multiclass_logistic_regression_model_bag_of_words_full(train_x, train_y)
    eval_model(log_reg_bow, test_x, test_y)
    rand_for_bow = _get_random_forest_model_bag_of_words_full(train_x, train_y)
    eval_model(rand_for_bow, test_x, test_y)
    nb_bow = _get_naive_bayes_model_bag_of_words_full(train_x, train_y)
    eval_model(nb_bow, test_x.A, test_y)

    train_x, train_y, test_x, test_y = tfidf_no_empty(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    log_reg_bow = _get_multiclass_logistic_regression_model_tfidf(train_x, train_y)
    eval_model(log_reg_bow, test_x, test_y)
    rand_for_bow = _get_random_forest_model_tfidf(train_x, train_y)
    print("random forest, tfidf")
    eval_model(rand_for_bow, test_x, test_y)
    nb_bow = _get_naive_bayes_model_tfidf(train_x.A, train_y)
    print("naive bayes, tfidf")
    eval_model(nb_bow, test_x.A, test_y)

    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)
    log_reg_bow = _get_multiclass_logistic_regression_model_doc2vec_simple(train_x, train_y)
    print("bow, doc2vec")
    eval_model(log_reg_bow, test_x, test_y)
    rand_for_bow = _get_random_forest_model_doc2vec_simple(train_x, train_y)
    print("random forest, doc2evc")
    eval_model(rand_for_bow, test_x, test_y)
    nb_bow = _get_naive_bayes_model_doc2vec_simple(train_x, train_y)
    print("naive bayes, doc2vec")
    eval_model(nb_bow, test_x, test_y)

# print average precision, recall, and f1 score for non-neural network models
def eval_model(model, test_x, test_y):
    pred = model.model.predict(test_x)
    pred_y = pd.DataFrame(pred, columns=['ON WG IDENTIFIER'])
    pred_y['ON WG IDENTIFIER'] = pred_y['ON WG IDENTIFIER'].astype(int)

    y = test_y.reset_index().drop('index', axis=1)
    labels = y['ON WG IDENTIFIER'].unique()

    precisions = []
    for label in labels:
        binary_y = y['ON WG IDENTIFIER'] == label
        binary_pred = pred_y['ON WG IDENTIFIER'] == label
        amount = len(binary_y[binary_y == True])
        p = average_precision_score(binary_y, binary_pred)
        precisions.append((p, amount))

    average_precision = 0
    average_amount = 0
    weighted_precision = 0
    weighted_amount = 0
    for precision, amount in precisions:
        average_precision += precision
        average_amount += 1
        weighted_precision += (precision * amount)
        weighted_amount += amount
    average_precision = average_precision / average_amount
    weighted_precision = weighted_precision / weighted_amount

    recalls = []
    for label in labels:
        binary_y = y['ON WG IDENTIFIER'] == label
        binary_pred = pred_y['ON WG IDENTIFIER'] == label
        amount = len(binary_y[binary_y == True])
        r = recall_score(binary_y, binary_pred)
        recalls.append((r, amount))

    average_recall = 0
    average_amount_recall = 0
    weighted_recall = 0
    weighted_amount_recall = 0
    for recall, amount in recalls:
        average_recall += recall
        average_amount_recall += 1
        weighted_recall += (recall * amount)
        weighted_amount_recall += amount
    average_recall = average_recall / average_amount_recall
    weighted_recall = weighted_recall / weighted_amount_recall

    i = 0
    f1s = []
    while i < len(recalls):
        recall = recalls[i][0]
        precision = precisions[i][0]
        amount = recalls[i][1]
        f1 = 2 * ((recall * precision) / (recall + precision))
        f1s.append((f1, amount))
        i += 1

    average_f1 = 0
    average_amount_f1 = 0
    weighted_f1 = 0
    weighted_amount_f1 = 0
    for f1, amount in f1s:
        average_f1 += f1
        average_amount_f1 += 1
        weighted_f1 += (f1 * amount)
        weighted_amount_f1 += amount
    average_f1 = average_f1 / average_amount_f1
    weighted_f1 = weighted_f1 / weighted_amount_f1

    # Average is equivelent to macro average, weighted average is equivelent to micro average
    print("Macro Precision: " + str(average_precision))
    print("Micro Precision: " + str(weighted_precision))
    print("Macro Recall: " + str(average_recall))
    print("Micro Recall: " + str(weighted_recall))
    print("Macro F1 Score: " + str(average_f1))
    print("Micro F1 Score: " + str(weighted_f1))

# print average precision, recall, and f1 score for a neural network
def eval_nn(model, test_x, test_y):
    pred = model.model.predict(test_x)
    pred = model.encoder.inverse_transform(pred)
    pred_y = pd.DataFrame(pred, columns=['ON WG IDENTIFIER'])
    pred_y['ON WG IDENTIFIER'] = pred_y['ON WG IDENTIFIER'].astype(int)

    y = test_y.reset_index().drop('index', axis=1)
    labels = y['ON WG IDENTIFIER'].unique()

    precisions = []
    for label in labels:
        binary_y = y['ON WG IDENTIFIER'] == label
        binary_pred = pred_y['ON WG IDENTIFIER'] == label
        amount = len(binary_y[binary_y == True])
        p = average_precision_score(binary_y, binary_pred)
        precisions.append((p, amount))

    average_precision = 0
    average_amount = 0
    weighted_precision = 0
    weighted_amount = 0
    for precision, amount in precisions:
        average_precision += precision
        average_amount += 1
        weighted_precision += (precision * amount)
        weighted_amount += amount
    average_precision = average_precision / average_amount
    weighted_precision = weighted_precision / weighted_amount

    recalls = []
    for label in labels:
        binary_y = y['ON WG IDENTIFIER'] == label
        binary_pred = pred_y['ON WG IDENTIFIER'] == label
        amount = len(binary_y[binary_y == True])
        r = recall_score(binary_y, binary_pred)
        recalls.append((r, amount))

    average_recall = 0
    average_amount_recall = 0
    weighted_recall = 0
    weighted_amount_recall = 0
    for recall, amount in recalls:
        average_recall += recall
        average_amount_recall += 1
        weighted_recall += (recall * amount)
        weighted_amount_recall += amount
    average_recall = average_recall / average_amount_recall
    weighted_recall = weighted_recall / weighted_amount_recall

    i = 0
    f1s = []
    while i < len(recalls):
        recall = recalls[i][0]
        precision = precisions[i][0]
        amount = recalls[i][1]
        f1 = 2 * ((recall * precision) / (recall + precision))
        f1s.append((f1, amount))
        i += 1

    average_f1 = 0
    average_amount_f1 = 0
    weighted_f1 = 0
    weighted_amount_f1 = 0
    for f1, amount in f1s:
        average_f1 += f1
        average_amount_f1 += 1
        weighted_f1 += (f1 * amount)
        weighted_amount_f1 += amount
    average_f1 = average_f1 / average_amount_f1
    weighted_f1 = weighted_f1 / weighted_amount_f1

    # Average is equivelent to macro average, weighted average is equivelent to micro average
    print("Macro Precision: " + str(average_precision))
    print("Micro Precision: " + str(weighted_precision))
    print("Macro Recall: " + str(average_recall))
    print("Micro Recall: " + str(weighted_recall))
    print("Macro F1 Score: " + str(average_f1))
    print("Micro F1 Score: " + str(weighted_f1))

def supervised_scratch():
    from Models.neural_net import MultiClassNNScratch
    from demo import get_bag_of_words
    from keras.models import load_model
    from Models.neural_net import top_3_accuracy

    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y, feature_names = get_bag_of_words(train_x_raw, train_y_raw,
                                                                                    val_x_raw, val_y_raw,
                                                                                    test_x_raw, test_y_raw)
    labels = data_reader.get_region_labels()['Code']

    model = MultiClassNNScratch(train_x.shape, np.array(labels), epochs=150, batch_size=1024)
    model.set_train_data(train_x, train_y)
    model.model = load_model('demo_nn.h5', custom_objects={'top_3_accuracy': top_3_accuracy})



    pred = model.model.predict(test_x)
    fpr = []
    tpr = []
    thr = []
    for label in labels:
        label_test_y = test_y['ON WG IDENTIFIER'] == label
        one_hot_label = model.encoder.transform([[label]])

        probs = pred[:, one_hot_label.A[0].argmax()]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(label_test_y, probs)
        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)
        thr.append(thresholds)

    # from IPython import embed
    # embed()

    plt.title('Receiver Operating Characteristic')

    for i in range(len(fpr)):
        fpr_c = fpr[i]
        tpr_c = tpr[i]

        plt.plot(fpr_c, tpr_c, 'b', marker='o')

    from IPython import embed
    embed()


    # plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig('nn_roc.png')

def self_training_site_split():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    all_tokens, _ = tokenize(df, df, save_missing_feature_as_string=False, remove_empty=True)
    _, _, vocab = tokens_to_bagofwords(all_tokens, all_tokens)

    lst = []
    from random import shuffle

    for i in df['src_file'].unique():
        lst.append(df[df['src_file'] == i])

    from Models.neural_net import MultiClassNNScratch

    model = MultiClassNNScratch((0, len(vocab)), np.array(data_reader.get_region_labels()['Code']), epochs=100,
                                batch_size=256)
    model.model.save_weights('empty_model.h5')
    shuffle(lst)

    file = open("output_dir/" + randomword(7) + '.txt', "w")
    train_set = lst[:15]
    test_set = lst[15:]

    item = pd.concat(train_set)
    train_x_raw, train_y_raw = prep_single_test_set(item)
    train_tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False,
                                         remove_empty=True)
    train_x_raw, train_y_raw, _ = tokens_to_bagofwords(train_tokens, train_y_raw, feature_names=vocab)


    test_x_raw, test_y_raw = get_x_y_split(pd.concat(test_set))
    test_tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False,
                                       remove_empty=True)
    test_x, test_y, _ = tokens_to_bagofwords(test_tokens, test_y_raw, feature_names=vocab)

    unlabelled_df = data_reader.get_east_dir()
    unlabelled_df = normalize_east_dir_df(unlabelled_df)

    # set up unlabelled data as semi-supervised data
    tokens, _ = tokenize(unlabelled_df, _, save_missing_feature_as_string=False, remove_empty=True)
    semi_x_base, _, _ = tokens_to_bagofwords(tokens, _, feature_names=vocab)

    # Confidence threshold to train on
    train_threshold = 0.6

    from scipy.sparse import vstack

    # from IPython import embed
    # embed()


    for i in range(30):
        i = 1
        model.model.load_weights('empty_model.h5')

        train_x, val_x = train_test_split(train_x_raw, random_state=1337, test_size=0.15, shuffle=True)
        train_y, val_y = train_test_split(train_y_raw, random_state=1337, test_size=0.15, shuffle=True)


        model.set_train_data(train_x, train_y)
        model.train(val_x, val_y)

        accuracy = evaluate_model_nn(model, test_x, test_y, plot_roc=False)

        pred = model.model.predict(semi_x_base)
        # convert probablities to 1 hot encoded output
        semi_y = np.zeros_like(pred)
        semi_y[np.arange(len(pred)), pred.argmax(1)] = 1
        # filter semi_x and semi_y to only include predictions above train_threshold
        semi_y = semi_y[pred.max(axis=1) > train_threshold]
        semi_x = semi_x_base[pred.max(axis=1) > train_threshold]

        semi_y = model.encoder.inverse_transform(semi_y)
        train_x_raw = vstack([train_x_raw, semi_x])
        train_y_raw = pd.concat([train_y_raw, pd.DataFrame(semi_y.ravel(), columns=['ON WG IDENTIFIER'])]).astype(int)

        file.write("%d, %4.2f, %d\n" % (i, accuracy, semi_x.shape[0]))


        i+=1
    file.close()

def per_site_accuracy_increase():
    data_reader = DataReader()
    df = data_reader.get_all_data()
    all_tokens, _ = tokenize(df, df, save_missing_feature_as_string=False, remove_empty=True)
    _, _, vocab = tokens_to_bagofwords(all_tokens, all_tokens)

    # from IPython import embed
    # embed()
    # tx, ty, vx, vy, yx, yy = get_train_validate_test_split(df)
    # _,_,_,_,_,_, vocab = bag_of_words_full_no_empty_val(tx, ty, vx, vy, yx, yy)

    lst = []
    from random import shuffle

    for i in df['src_file'].unique():
        lst.append(df[df['src_file'] == i])




    # tokens, _ = tokenize(df, df, save_missing_feature_as_string=False, remove_empty=True)
    # _, _, vocab = tokens_to_bagofwords(tokens, _)


    # train_x_raw, train_y_raw = get_x_y_split(pd.concat(train_set))


    from Models.neural_net import MultiClassNNScratch

    model = MultiClassNNScratch((0, len(vocab)), np.array(data_reader.get_region_labels()['Code']), epochs=100,
                                batch_size=256)
    model.model.save_weights('empty_model.h5')

    # from IPython import embed
    # embed()

    for i in range(30):

        shuffle(lst)

        i = 1
        file = open("output_dir/" + randomword(7) + '.txt', "w")
        while i < len(lst):
            model.model.load_weights('empty_model.h5')
            train_set = lst[:i]
            test_set = lst[i:]

            test_x_raw, test_y_raw = get_x_y_split(pd.concat(test_set))
            test_tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False,
                                               remove_empty=True)
            test_x, test_y, _ = tokens_to_bagofwords(test_tokens, test_y_raw, feature_names=vocab)

            item = pd.concat(train_set)
            train_x_raw, train_y_raw, val_x_raw, val_y_raw = get_train_test_split(item)
            train_tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_empty=True)
            train_x, train_y, _ = tokens_to_bagofwords(train_tokens, train_y_raw, feature_names=vocab)

            val_tokens, val_y_raw = tokenize(val_x_raw, val_y_raw, save_missing_feature_as_string=False, remove_empty=True)
            val_x, val_y, _ = tokens_to_bagofwords(val_tokens, val_y_raw, feature_names=vocab)

            model.set_train_data(train_x, train_y)
            model.train(val_x, val_y)

            accuracy = evaluate_model_nn(model, test_x, test_y, plot_roc=False)
            file.write("%d, %d, %4.2f, %d"%(len(train_set), len(test_set), accuracy, len(item)))

            i+=1
        file.close()
    # from IPython import embed
    # embed()

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def per_site_analysis():
    data_reader = DataReader()
    df = data_reader.get_all_data()


    lst = []

    for i in df['src_file'].unique():
        lst.append(df[df['src_file'] == i])

    accuracies = []

    for i in range(len(lst)):
        test_set = lst[i]
        from random import randint
        val_idx = randint(0, len(lst)-2)
        if val_idx >= i:
            val_idx +=1
        validation_set = lst[val_idx]
        train_set = lst[:i] + lst[i+1:]

        test_x_raw, test_y_raw = get_x_y_split(test_set)
        val_x_raw, val_y_raw = get_x_y_split(validation_set)
        train_x_raw, train_y_raw = get_x_y_split(pd.concat(train_set))


        train_x, train_y, val_x, val_y, test_x, test_y = bag_of_words_full_no_empty_val(train_x_raw, train_y_raw,
                                                                  val_x_raw, val_y_raw, test_x_raw,
                                                                  test_y_raw)

        model = _get_nn_model_bag_of_words_simple_scratch(train_x, train_y, val_x, val_y,
                                                          data_reader.get_region_labels()['Code'], epochs=100,
                                                          batch_size=256)


        accuracies.append(evaluate_model_nn(model, test_x, test_y, plot_roc=False))

    from IPython import embed
    embed()





def run_models_on_autoencoded_bagofwords():
    from Models.logistic_regression import MultiClassLogisticRegression
    from Models.random_forest import RandomForest
    from Models.naive_bayes import NaiveBayes
    from Models.svm import SVM

    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y = bag_of_words_full_no_empty_val_no_num_no_short_no_repeat(
        train_x_raw, train_y_raw,
        val_x_raw, val_y_raw, test_x_raw,
        test_y_raw)

    encoder = get_encoder(train_x, test_x, 4096)
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)
    encoded_val = encoder.predict(val_x)

    print('neural net ae')
    model = _get_nn_model_bag_of_words_simple_scratch(encoded_train, train_y, encoded_val, val_y,
                                                      data_reader.get_region_labels()['Code'], epochs=100,
                                                      batch_size=256)
    evaluate_model_nn(model, encoded_test, test_y, plot_roc=False)

    print('logistic regression ae')
    model = MultiClassLogisticRegression()
    model.train(encoded_train, train_y)
    evaluate_model(model, encoded_test, test_y, plot_roc=False)

    print('random forest ae')
    model = RandomForest()
    model.train(encoded_train, train_y)
    evaluate_model(model, encoded_test, test_y, plot_roc=False)

    print('naive bayes ae')
    model = NaiveBayes()
    model.train(encoded_train, train_y)
    evaluate_model(model, encoded_test, test_y, plot_roc=False)

    print('svm ae')
    model = SVM()
    model.train(encoded_train, train_y)
    evaluate_model(model, encoded_test, test_y, plot_roc=False)

def run_all_models():
    from Models.logistic_regression import MultiClassLogisticRegression
    from Models.naive_bayes import NaiveBayes
    from Models.naive_bayes import MultinomialNaiveBayes
    from Models.random_forest import RandomForest
    from Models.svm import SVM
    from Models.neural_net import MultiClassNNScratch
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)


    tokenizers = [('repeats, shorts, nums',(lambda x, y: tokenize(x, y, remove_repeats=True, remove_short=True, remove_empty=True, remove_num=True))),
                 ('repeats, shorts, no nums',(lambda x, y: tokenize(x, y, remove_repeats=True, remove_short=True, remove_empty=True, remove_num=False))),
                 ('repeats, no shorts, nums',(lambda x, y: tokenize(x, y, remove_repeats=True, remove_short=False, remove_empty=True, remove_num=True))),
                 ('repeats, no shorts, no nums',(lambda x, y: tokenize(x, y, remove_repeats=True, remove_short=False, remove_empty=True, remove_num=False))),
                 ('no repeats, shorts, nums',(lambda x, y: tokenize(x, y, remove_repeats=False, remove_short=True, remove_empty=True,remove_num=True))),
                 ('no repeats, shorts, no nums',(lambda x, y: tokenize(x, y, remove_repeats=False, remove_short=True, remove_empty=True,remove_num=False))),
                 ('no repeats, no shorts, nums',(lambda x, y: tokenize(x, y, remove_repeats=False, remove_short=False, remove_empty=True, remove_num=True))),
                 ('no repeats, no shorts, no nums',(lambda x, y: tokenize(x, y, remove_repeats=False, remove_short=False, remove_empty=True, remove_num=False)))]
    embeddings = [('doc2vec_1024',(lambda x,y,model: tokens_to_doc2vec(x, y, model=model, vector_size=1024))),
                  ('doc2vec_4196', (lambda x, y, model: tokens_to_doc2vec(x, y, model=model, vector_size=4196))),
                  ('doc2vec_16384',(lambda x,y,model: tokens_to_doc2vec(x, y, model=model, vector_size=16384))),
                  ('bagofwords',(lambda x,y, feature_names: tokens_to_bagofwords(x, y, vectorizer_class=CountVectorizer, feature_names=feature_names))),
                  ('tfidf',(lambda x, y, feature_names: tokens_to_bagofwords(x, y, vectorizer_class=TfidfVectorizer, feature_names=feature_names)))]

    models = [('logistic_regression', MultiClassLogisticRegression), ('naive_bayes', NaiveBayes), ('multinomial_naive_bayes', MultinomialNaiveBayes), ('random_forest', RandomForest), ('svm', SVM), ('nn', MultiClassNNScratch)]

    for tokenizer_name, tokenizer in tokenizers:
        train_x, train_y = tokenizer(train_x_raw, train_y_raw)
        val_x, val_y = tokenizer(val_x_raw, val_y_raw)
        test_x, test_y = tokenizer(test_x_raw, test_y_raw)
        for embedding_name, embedding in embeddings:
            emb_train_x, emb_train_y, pass_on = embedding(train_x, train_y, None)
            emb_val_x, emb_val_y, _ = embedding(val_x, val_y, pass_on)
            emb_test_x, emb_test_y, _ = embedding(test_x, test_y, pass_on)
            for model_name, the_model in models:
                if model_name == 'nn':
                    print('training: ' + tokenizer_name + ' ' + embedding_name + ' ' + model_name)
                    model = the_model(train_x.shape, np.array(data_reader.get_region_labels()['Code']), epochs=100,
                                                    batch_size=256)
                    model.set_train_data(train_x, train_y)
                    model.train(val_x, val_y)
                    print(tokenizer_name + ' ' + embedding_name + ' ' + model_name)
                    evaluate_model_nn(model, test_x, test_y, plot_roc=False)
                else:
                    print('training: ' + tokenizer_name + ' ' + embedding_name + ' ' + model_name)
                    model = the_model()
                    model.train(train_x, train_y)
                    print(tokenizer_name + ' ' + embedding_name + ' ' + model_name)
                    evaluate_model(model, test_x, test_y, plot_roc=False)



def autoencoder_tsne():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y = bag_of_words_full_no_empty_val_no_num_no_short_no_repeat(train_x_raw, train_y_raw,
                                                                                    val_x_raw, val_y_raw, test_x_raw,
                                                                                    test_y_raw)

    encoder = get_encoder(train_x, test_x, int(len(data_reader.get_region_labels()['Code'])))
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)
    encoded_val = encoder.predict(val_x)

    from IPython import embed
    embed()

    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(encoded_test)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(pca_result)

    test_y_shift = test_y - test_y.min()

    y_test_cat = np_utils.to_categorical(test_y_shift, num_classes=int(test_y_shift.max()) +1)

    import matplotlib.pyplot as plt

    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(10, 10))
    for cl in range(data_reader.get_region_labels()['Code']):
        indices = np.where(test_y == cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=cl)
    plt.legend()
    plt.savefig('tsne.png')
    plt.show()

def auto_encoder_and_nn():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y = tfidf_no_empty_val(train_x_raw, train_y_raw,
                                                                                    val_x_raw, val_y_raw, test_x_raw,
                                                                                    test_y_raw)

    encoder, decoder = get_encoder(train_x, test_x, 4096)
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)
    encoded_val = encoder.predict(val_x)

    from IPython import embed
    embed()
    model = _get_nn_model_bag_of_words_simple_scratch(encoded_train, train_y, encoded_val, val_y,
                                                      data_reader.get_region_labels()['Code'], epochs=100,
                                                      batch_size=256)

    evaluate_model_nn(model, encoded_test, test_y, plot_roc=False)


def neural_net_scratch_bag_of_words(function):
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)

    train_x, train_y, val_x, val_y, test_x, test_y = function(train_x_raw, train_y_raw,
                                                                                    val_x_raw, val_y_raw, test_x_raw,
                                                                                    test_y_raw)

    model = _get_nn_model_bag_of_words_simple_scratch(train_x, train_y, val_x, val_y,
                                                      data_reader.get_region_labels()['Code'], epochs=100,
                                                      batch_size=128)

    from IPython import embed
    embed()

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

# 80 percent with bag of words, full on _get_nn_model_bag_of_words_simple_v2 (single layer dense)
# 80 percent with bag of words full on _get_nn_model_bag_of_words_simple (multi layer dense)
def good_neural_net():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    # unlabelled_df = data_reader.get_east_dir()
    # unlabelled_df = normalize_east_dir_df(unlabelled_df)

    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    train_x, train_y, test_x, test_y = bag_of_words_full_no_empty(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    model  = _get_nn_model_bag_of_words_simple_v2(train_x, train_y, data_reader.get_region_labels()['Code'],
                                                      epochs=50, batch_size=64)

    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

    # tokens_train, _ = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False, remove_empty=True)
    # _, _, feature_names = tokens_to_bagofwords(tokens_train, _)
    #
    # tokens, _ = tokenize(unlabelled_df, _, save_missing_feature_as_string=False, remove_empty=True)
    # semi_x_base, _, _ = tokens_to_bagofwords(tokens, _, feature_names=feature_names)
    # train_threshold = 0.9
    #
    # from IPython import embed
    # embed()
    #
    # for i in range(30):
    #     m = model.model
    #     pred = m.predict(semi_x_base)
    #     semi_y = np.zeros_like(pred)
    #     semi_y[np.arange(len(pred)), pred.argmax(1)] = 1
    #     semi_y = semi_y[pred.max(axis=1) > train_threshold]
    #     semi_x = semi_x_base[pred.max(axis=1) > train_threshold]
    #
    #     m.fit(semi_x, semi_y, batch_size=64, epochs=100)
    #     m.fit(train_x, model.encoder.transform(train_y), batch_size=32, epochs=10)
    #
    #     evaluate_model_nn(model, test_x, test_y, plot_roc=False)
    #     semi_x_base = semi_x_base[~(pred.max(axis=1) > train_threshold)]
    #
    # from IPython import embed
    # embed()

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

    return score

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