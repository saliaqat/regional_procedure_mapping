from data_manipulator import *
from cache_em_all import Cachable

# Note: All functions with Cacheable before them cache the results the first time run, and read from the disk
# every subsequent call. Please don't change them, but feel free to use them. You can add functions if you need
# different functionality, and if it takes long to run, consider adding the Cacheable annotation before the function
# to cache the return values.

@Cachable("simple_bag_of_words.pkl", version=1)
def bag_of_words_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, feature_names = tokens_to_bagofwords(tokens, train_y_raw)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y, _ = tokens_to_bagofwords(tokens, test_y_raw, feature_names=feature_names)

    return train_x, train_y, test_x, test_y


@Cachable("simple_doc2vec.pkl", version=1)
def doc2vec_simple(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model)

    return train_x, train_y, test_x, test_y

@Cachable("save_missing_features_remove_repeats_remove_short_doc2vec.pkl", version=1)
def doc2vec_save_missing_features_remove_repeats_remove_short(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                           save_missing_feature_as_string=False,
                                           remove_repeats=True, remove_short=True)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw)

    tokens, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                          save_missing_feature_as_string=False,
                                          remove_repeats=True, remove_short=True)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model)

    return train_x, train_y, test_x, test_y

@Cachable("simple_doc2vec_4096.pkl", version=1)
def doc2vec_simple_4096(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=4096)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=4096)

    return train_x, train_y, test_x, test_y

@Cachable("save_missing_features_remove_repeats_remove_short_doc2vec_4096.pkl", version=1)
def doc2vec_save_missing_features_remove_repeats_remove_short_4096(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                           save_missing_feature_as_string=False,
                                           remove_repeats=True, remove_short=True)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=4096)

    tokens, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                          save_missing_feature_as_string=False,
                                          remove_repeats=True, remove_short=True)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=4096)

    return train_x, train_y, test_x, test_y

@Cachable("simple_doc2vec_8192.pkl", version=1)
def doc2vec_simple_8192(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=8192)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=8192)

    return train_x, train_y, test_x, test_y

@Cachable("save_missing_features_remove_repeats_remove_short_doc2vec_8192.pkl", version=1)
def doc2vec_save_missing_features_remove_repeats_remove_short_8192(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                           save_missing_feature_as_string=False,
                                           remove_repeats=True, remove_short=True)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=8192)

    tokens, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                          save_missing_feature_as_string=False,
                                          remove_repeats=True, remove_short=True)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=8192)

    return train_x, train_y, test_x, test_y

@Cachable("simple_doc2vec_16384.pkl", version=1)
def doc2vec_simple_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, save_missing_feature_as_string=False)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=16384)

    tokens, test_y_raw = tokenize(test_x_raw, test_y_raw, save_missing_feature_as_string=False)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=16384)

    return train_x, train_y, test_x, test_y

@Cachable("save_missing_features_remove_repeats_remove_short_doc2vec_16384.pkl", version=1)
def doc2vec_save_missing_features_remove_repeats_remove_short_16384(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    tokens, train_y_raw = tokenize_columns(train_x_raw, train_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                           save_missing_feature_as_string=False,
                                           remove_repeats=True, remove_short=True)
    train_x, train_y, doc2vec_model = tokens_to_doc2vec(tokens, train_y_raw, vector_size=16384)

    tokens, test_y_raw = tokenize_columns(test_x_raw, test_y_raw, regex_string=r'[a-zA-Z0-9/]+',
                                          save_missing_feature_as_string=False,
                                          remove_repeats=True, remove_short=True)
    test_x, test_y = tokens_to_doc2vec(tokens, test_y_raw, model=doc2vec_model, vector_size=16384)

    return train_x, train_y, test_x, test_y


