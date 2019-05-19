from Models.logistic_regression import BinaryLogisticRegressionModel
from Models.model import Model
from data_manipulator_interface import bag_of_words_full_no_empty, tfidf_no_empty, doc2vec_simple, bag_of_words_full_no_empty_val_no_num_no_short_no_repeat
from data_reader import DataReader
from data_manipulator import *
from cached_models import _get_nn_model_bag_of_words_simple_scratch
from cached_models import _get_random_forest_model_bag_of_words_full
from cached_models import _get_naive_bayes_model_bag_of_words_full
from cached_models import _get_multiclass_logistic_regression_model_tfidf
from cached_models import _get_random_forest_model_tfidf
from cached_models import _get_naive_bayes_model_tfidf
from cached_models import _get_multiclass_logistic_regression_model_doc2vec_simple
from cached_models import _get_random_forest_model_doc2vec_simple
from cached_models import _get_naive_bayes_model_doc2vec_simple
from cached_models import _get_multiclass_logistic_regression_model_bag_of_words_full
from supervised_methods import eval_nn, evaluate_model_nn, eval_model, evaluate_model, randomword
from run_autoencoder import get_encoder

import pandas as pd

# Code to get evaluation results
def main():
    # Calculates the qualitative method evaluation.
    # Evaluate logistic regression, random forest, naive bayes and neural network on a pub med word2vec representation
    eval_pub_med()
    # Evaluate logistic regression, random forest, naive bayes and neural network on a autoencoder + bagofwords representation
    eval_ae()
    # Evaluate logistic regression, random forest, naive bayes and neural network on a bag of words, tfidf and doc2vec represetnation
    eval()

    # Runs the generalizability check on nn and bow
    # Results saved to a set of files in output_dir. Files will have random name of 7 characters.
    per_site_accuracy_increase()



    pass

def eval_pub_med():
    from gensim.models.keyedvectors import KeyedVectors
    # Need to download file from http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin
    # Load the pubmed model
    model = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
    # Load data into train/validate/test sets
    data_reader = DataReader()
    df = data_reader.get_all_data()
    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    tokens_train, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False,
                                       remove_empty=True)
    # for the each tokenized vector in the train set, run the model on each word and take the average.
    # If no words are vectorized by pubmed, append an 0 vector
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

    # run the same for the validation set
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

    # run the same for the test set
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

    # train the neural network model and calculate the precision, recall, f1 score, and accuracy
    print("pubmed, nn")
    nn_model = _get_nn_model_bag_of_words_simple_scratch(pub_med_train, train_y_raw, pub_med_val, val_y_raw,
                                                         data_reader.get_region_labels()['Code'], epochs=100, batch_size=256)
    eval_nn(nn_model, pub_med_test, test_y_raw)
    evaluate_model_nn(nn_model, pub_med_test, test_y_raw, plot_roc=False)
    # train the logistic regression model and calculate the precision, recall, f1 score, and accuracy
    print("pubmed, logistic regression")
    from Models.logistic_regression import MultiClassLogisticRegression
    log_reg = MultiClassLogisticRegression()
    log_reg.train(pub_med_train, train_y_raw)
    eval_model(log_reg, pub_med_test, test_y_raw)
    evaluate_model(log_reg, pub_med_test, test_y_raw, plot_roc=False)
    # train the random forest model and calculate the precision, recall, f1 score, and accuracy
    print("pubmed, random forest")
    from Models.random_forest import RandomForest
    rand_for = RandomForest()
    rand_for.train(pub_med_train, train_y_raw)
    eval_model(rand_for, pub_med_test, test_y_raw)
    evaluate_model(rand_for, pub_med_test, test_y_raw, plot_roc=False)
    # train the naive bayes model and calculate the precision, recall, f1 score, and accuracy
    print("pubmed, naivebayes")
    from Models.naive_bayes import NaiveBayes
    nb = NaiveBayes()
    nb.train(pub_med_train, train_y_raw)
    eval_model(nb, pub_med_test, test_y_raw)
    evaluate_model(nb, pub_med_test, test_y_raw, plot_roc=False)

def eval_ae():
    from Models.logistic_regression import MultiClassLogisticRegression
    from Models.random_forest import RandomForest
    from Models.naive_bayes import NaiveBayes
    from Models.svm import SVM
    # load data
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, val_x_raw, val_y_raw, test_x_raw, test_y_raw = get_train_validate_test_split(df)
    train_x, train_y, val_x, val_y, test_x, test_y = bag_of_words_full_no_empty_val_no_num_no_short_no_repeat(
        train_x_raw, train_y_raw,
        val_x_raw, val_y_raw, test_x_raw,
        test_y_raw)
    # Train an auto encoder of size 4096
    encoder = get_encoder(train_x, test_x, 4096)
    # use auto encoder to encode the train, validate and test sets
    encoded_train = encoder.predict(train_x)
    encoded_test = encoder.predict(test_x)
    encoded_val = encoder.predict(val_x)

    # train the neural network model and calculate the precision, recall, f1 score, and accuracy
    print('neural net ae')
    model = _get_nn_model_bag_of_words_simple_scratch(encoded_train, train_y, encoded_val, val_y,
                                                      data_reader.get_region_labels()['Code'], epochs=100,
                                                      batch_size=256)
    eval_nn(model, encoded_test, test_y)
    evaluate_model_nn(model, encoded_test, test_y)
    # train the logistic regression model and calculate the precision, recall, f1 score, and accuracy
    print('logistic regression ae')
    model = MultiClassLogisticRegression()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

    # train the random forest model and calculate the precision, recall, f1 score, and accuracy
    print('random forest ae')
    model = RandomForest()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

    # train the naive bayes model and calculate the precision, recall, f1 score, and accuracy
    print('naive bayes ae')
    model = NaiveBayes()
    model.train(encoded_train, train_y)
    model_obj = lambda: None
    model_obj.model = model
    eval_model(model_obj, encoded_test, test_y)
    evaluate_model(model, encoded_test, test_y)

def eval():
    # load data
    data_reader = DataReader()
    df = data_reader.get_all_data()

    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    # get bag of words
    train_x, train_y, test_x, test_y = bag_of_words_full_no_empty(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    #train logistic regression, random forest and naive bayes on bag of words, and run get accuracy, precision, recall, f1score
    log_reg_bow = _get_multiclass_logistic_regression_model_bag_of_words_full(train_x, train_y)
    eval_model(log_reg_bow, test_x, test_y)
    evaluate_model(log_reg_bow, test_x, test_y, plot_roc=False)
    rand_for_bow = _get_random_forest_model_bag_of_words_full(train_x, train_y)
    eval_model(rand_for_bow, test_x, test_y)
    evaluate_model(rand_for_bow, test_x, test_y, plot_roc=False)
    nb_bow = _get_naive_bayes_model_bag_of_words_full(train_x, train_y)
    eval_model(nb_bow, test_x.A, test_y)
    evaluate_model(nb_bow, test_x.A, test_y, plot_roc=False)

    # get tfidf
    train_x, train_y, test_x, test_y = tfidf_no_empty(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    # train logistic regression, random forest and naive bayes on tfidf, and run get accuracy, precision, recall, f1score
    log_reg_bow = _get_multiclass_logistic_regression_model_tfidf(train_x, train_y)
    eval_model(log_reg_bow, test_x, test_y)
    evaluate_model(log_reg_bow, test_x, test_y, plot_roc=False)
    rand_for_bow = _get_random_forest_model_tfidf(train_x, train_y)
    print("random forest, tfidf")
    eval_model(rand_for_bow, test_x, test_y)
    evaluate_model(rand_for_bow, test_x, test_y, plot_roc=False)
    nb_bow = _get_naive_bayes_model_tfidf(train_x.A, train_y)
    print("naive bayes, tfidf")
    eval_model(nb_bow, test_x.A, test_y)
    evaluate_model(nb_bow, test_x, test_y, plot_roc=False)

    # get doc2vec
    train_x, train_y, test_x, test_y = doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw)

    # train logistic regression, random forest and naive bayes on doc2vec, and run get accuracy, precision, recall, f1score
    log_reg_bow = _get_multiclass_logistic_regression_model_doc2vec_simple(train_x, train_y)
    print("bow, doc2vec")
    eval_model(log_reg_bow, test_x, test_y)
    evaluate_model(log_reg_bow, test_x, test_y, plot_roc=False)
    rand_for_bow = _get_random_forest_model_doc2vec_simple(train_x, train_y)
    print("random forest, doc2evc")
    eval_model(rand_for_bow, test_x, test_y)
    evaluate_model(rand_for_bow, test_x, test_y, plot_roc=False)
    nb_bow = _get_naive_bayes_model_doc2vec_simple(train_x, train_y)
    print("naive bayes, doc2vec")
    eval_model(nb_bow, test_x, test_y)
    evaluate_model(nb_bow, test_x, test_y, plot_roc=False)

def per_site_accuracy_increase():
    # load data
    data_reader = DataReader()
    df = data_reader.get_all_data()
    all_tokens, _ = tokenize(df, df, save_missing_feature_as_string=False, remove_empty=True)
    _, _, vocab = tokens_to_bagofwords(all_tokens, all_tokens)

    lst = []
    from random import shuffle

    #split data on source hospital and save to seperate dataframes in a list
    for i in df['src_file'].unique():
        lst.append(df[df['src_file'] == i])

    from Models.neural_net import MultiClassNNScratch

    # save an empty neural network so we can quickly reset the network
    model = MultiClassNNScratch((0, len(vocab)), np.array(data_reader.get_region_labels()['Code']), epochs=100,
                                batch_size=256)
    model.model.save_weights('empty_model.h5')

    # run evaluation some n times
    for i in range(30):
        # shuffle the order
        shuffle(lst)
        # iterate from 1 to len(lst)-1 from size of train set. Train model on 1->i sites and test on i->len(lst)-1.
        # Print results to file so we can easily visualize later.
        # each run of the 30 gets its own file.
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


if __name__ == '__main__':
    main()