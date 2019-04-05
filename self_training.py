from data_reader import DataReader
from data_manipulator import *

from cached_models import _get_nn_model_bag_of_words_simple_v2
from supervised_methods import evaluate_model_nn

def main():
    data_reader = DataReader()
    df = data_reader.get_all_data()

    # random split of data
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = get_train_test_split(df)

    # set up train data
    train_tokens, train_y_raw = tokenize(train_x_raw, train_y_raw, save_missing_feature_as_string=False,
                                       remove_empty=True)
    train_x, train_y, feature_names = tokens_to_bagofwords(train_tokens, train_y_raw)

    # train model
    model  = _get_nn_model_bag_of_words_simple_v2(train_x, train_y, data_reader.get_region_labels()['Code'],
                                                      epochs=50, batch_size=64)

    # set up test data
    test_tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False, remove_empty=True)
    test_x, test_y, _ = tokens_to_bagofwords(test_tokens, test_y_raw, feature_names=feature_names)

    # evaluate model
    evaluate_model_nn(model, test_x, test_y, plot_roc=False)

    # ABOVE IS BASIC SUPERVISED LEARNING TO GENERATE MODEL
    #################################################
    # BELOW IS SEMI-SUPERVISED SELF-TRAINING TO FUTHER TRAIN MODEL

    # read unlabelled data and format it to be the same as labelled data
    unlabelled_df = data_reader.get_east_dir()
    unlabelled_df = normalize_east_dir_df(unlabelled_df)

    # set up unlabelled data as semi-supervised data
    tokens, _ = tokenize(unlabelled_df, _, save_missing_feature_as_string=False, remove_empty=True)
    semi_x_base, _, _ = tokens_to_bagofwords(tokens, _, feature_names=feature_names)

    # Confidence threshold to train on
    train_threshold = 0.8
    semi_train_amount = 30

    # SELF TRAIN MANY TIMES
    for i in range(semi_train_amount):
        # get predictions on unlabelled data
        pred = model.model.predict(semi_x_base)
        # convert probablities to 1 hot encoded output
        semi_y = np.zeros_like(pred)
        semi_y[np.arange(len(pred)), pred.argmax(1)] = 1
        # filter semi_x and semi_y to only include predictions above train_threshold
        semi_y = semi_y[pred.max(axis=1) > train_threshold]
        semi_x = semi_x_base[pred.max(axis=1) > train_threshold]

        # train on semi supervised data
        model.model.fit(semi_x, semi_y, batch_size=64, epochs=100)
        # retrain on original train data
        model.model.fit(train_x, model.encoder.transform(train_y), batch_size=32, epochs=10)

        # evaluate model
        evaluate_model_nn(model, test_x, test_y, plot_roc=False)

        # remove semi data used in this iteration from future iterations
        semi_x_base = semi_x_base[~(pred.max(axis=1) > train_threshold)]

if __name__ == '__main__':
    main()