from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np

from cache_em_all import Cachable

all_columns = ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION',
       'PACS SITE PROCEDURE CODE', 'PACS PROCEDURE DESCRIPTION',
       'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY',
       'DEFAULT LOCALIZATION FOR FEM', 'ON WG IDENTIFIER']
feature_columns = ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION', 'PACS SITE PROCEDURE CODE', 'PACS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY', 'DEFAULT LOCALIZATION FOR FEM']
feature_columns_with_desc = [('RIS PROCEDURE CODE', "risproccode"), ('RIS PROCEDURE DESCRIPTION', "risprocdesc"),
                   ('PACS SITE PROCEDURE CODE', "pacsproccode"), ('PACS PROCEDURE DESCRIPTION', "pacsprocdesc"),
                   ('PACS STUDY DESCRIPTION', "pacsstudydesc"), ('PACS BODY PART', "pacsbodypart"), ('PACS MODALITY', "pacsmodality"),
                   ('DEFAULT LOCALIZATION FOR FEM', 'default')]
text_columns = ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION', 'PACS SITE PROCEDURE CODE', 'PACS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY']
text_columns_with_desc = [('RIS PROCEDURE CODE', "risproccode"), ('RIS PROCEDURE DESCRIPTION', "risprocdesc"),
                   ('PACS SITE PROCEDURE CODE', "pacsproccode"), ('PACS PROCEDURE DESCRIPTION', "pacsprocdesc"),
                   ('PACS STUDY DESCRIPTION', "pacsstudydesc"), ('PACS BODY PART', "pacsbodypart"), ('PACS MODALITY', "pacsmodality")]
label_columns = ['ON WG IDENTIFIER']

def prep_single_test_set(input_df, random_state=1337, label=label_columns, test_size=0.25):
    # dropping src_file since its is not a feature
    df = input_df.drop('src_file', axis=1)

    train_x = df.drop(label, axis=1)
    train_y = df[label]

    return train_x, train_y

def get_train_test_split(input_df, random_state=1337, label=label_columns, test_size=0.25):
    # dropping src_file since its is not a feature
    df = input_df.drop('src_file', axis=1)

    train, test = train_test_split(df, random_state=random_state, test_size=test_size, shuffle=True)

    train_x = train.drop(label, axis=1)
    train_y = train[label]
    test_x = test.drop(label, axis=1)
    test_y = test[label]

    return train_x, train_y, test_x, test_y


def get_train_validate_test_split(input_df, random_state=1337, label=label_columns, test_size=0.125,
                                  validation_split=0.125):
    # dropping src_file since its is not a feature
    df = input_df.drop('src_file', axis=1)

    first_split = test_size + validation_split
    second_split = validation_split / first_split

    train, test_validation = train_test_split(df, random_state=random_state, test_size=first_split, shuffle=True)
    test, validation = train_test_split(test_validation, random_state=random_state, test_size=second_split,
                                        shuffle=True)

    train_x = train.drop(label, axis=1)
    train_y = train[label]
    validation_x = validation.drop(label, axis=1)
    validation_y = validation[label]
    test_x = test.drop(label, axis=1)
    test_y = test[label]

    return train_x, train_y, validation_x, validation_y, test_x, test_y

# def get_train_validate_test_split_remove_low_occurance_classes(input_df, random_state=1337, label=label_columns, test_size=0.125,
#                                   validation_split=0.125):
#     # dropping src_file since its is not a feature
#     df = input_df.drop('src_file', axis=1)
#
#     first_split = test_size + validation_split
#     second_split = validation_split / first_split
#
#     train, test_validation = train_test_split(df, random_state=random_state, test_size=first_split, shuffle=True)
#     test, validation = train_test_split(test_validation, random_state=random_state, test_size=second_split,
#                                         shuffle=True)
#
#     train_x = train.drop(label, axis=1)
#     train_y = train[label]
#     validation_x = validation.drop(label, axis=1)
#     validation_y = validation[label]
#     test_x = test.drop(label, axis=1)
#     test_y = test[label]
#
#     from IPython import embed
#     embed()
#
#     return train_x, train_y, validation_x, validation_y, test_x, test_y


# Takes as input features as x, and labels as y.
# Returns a pandas series containing the all features tokenized in a list for each row
# Tokenizer considers alphanumeric characters only (can specify using parameters)
# Tokenizer only lower cases the strings. Does no other processing.
# Optional parameter save_missing_as_feature sets all missing columns to a unique
# code to save that that column was missing. Default setting removes nans from tokens.
def tokenize(x, y, save_missing_feature_as_string=False, regex_string=r'[a-zA-Z0-9]+', remove_repeats=False,
             remove_short=False, remove_empty=False, remove_num=False):
    x = x[text_columns]
    x[text_columns] = x[text_columns].astype(str)
    tokenizer = RegexpTokenizer(regex_string)
    x['tokens'] =  [[]] * len(x)

    if save_missing_feature_as_string:
        for col in text_columns_with_desc:
            x[col[0]] = x[col[0]].apply(lambda x: x if x != 'nan' else ("missing" + col[1]))
    else:
        for col in text_columns_with_desc:
            x[col[0]] = x[col[0]].apply(lambda x: x if x != 'nan' else "")

    for col in text_columns:
        tokens = x[col].apply(lambda x: ((tokenizer.tokenize(x.lower()))))
        x['tokens'] = x['tokens'] + tokens

    if remove_repeats:
        x['tokens'] = x['tokens'].apply(lambda y: list(set(y)))
    if remove_short:
        x['tokens'] = x['tokens'].apply(lambda y: [z for z in y if len(z) > 1])
    if remove_num:
        x['tokens'] = x['tokens'].apply(lambda y: [z for z in y if not z.isdigit()])
    if remove_empty:
        if (len(x[x['tokens'].str.len() == 0]) != 0):
            y = y[x['tokens'].str.len() != 0]
            x = x[x['tokens'].str.len() != 0]


    return x['tokens'], y

def tokens_to_word2vec(x, y):
    model = Word2Vec(min_count=1)  # or w.e ur settings are
    model.build_vocab(x)
    model.train(x, total_examples=len(x), epochs=10)
    # print(model.wv['first'])
    from IPython import embed
    embed()

    # model = Word2Vec(x)
    # model.train(x, total_examples=len(x), epochs = 10)

def tokens_to_doc2vec(x, y, model=None, vector_size=512, min_count=1, workers=1):
    # from IPython import embed
    # embed()
    if model is None:
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x)]
        model = Doc2Vec(documents, vector_size=vector_size, min_count=min_count, workers=workers, epochs=10)
        represented_x = []

        for item in x:
            represented_x.append(model.infer_vector(item))
        return np.array(represented_x), y, model
    else:
        represented_x = []

        for item in x:
            represented_x.append(model.infer_vector(item))
        return np.array(represented_x), y, model

# Takes as input tokens, labels and a vectorizer and returns the vectorized tokens, labels and feature_names
# Vectorizer is defaulted to count vectorizer, but can be TfidfVectorizer or any other vectorizer
def tokens_to_bagofwords(x, y, vectorizer_class=CountVectorizer, feature_names=None):
    if feature_names:
        vectorizer = vectorizer_class(tokenizer=lambda x: x, lowercase=False, strip_accents=False, vocabulary=feature_names)
    else:
        vectorizer = vectorizer_class(tokenizer= lambda x: x, lowercase=False, strip_accents=False)
    weights = vectorizer.fit_transform(x)
    feature_names = vectorizer.get_feature_names()
    return weights, y, feature_names

# Only tokenize subset of data
def tokenize_columns(x, y, save_missing_feature_as_string=False, regex_string=r'[a-zA-Z0-9]+', remove_repeats=False,
                     remove_short=False, remove_empty=False, remove_num=False):
    columns = ['RIS PROCEDURE DESCRIPTION', 'PACS PROCEDURE DESCRIPTION', 'PACS STUDY DESCRIPTION', 'PACS BODY PART',
               'PACS MODALITY']
    columns_with_desc = [('RIS PROCEDURE DESCRIPTION', "risprocdesc"),
                         ('PACS PROCEDURE DESCRIPTION', "pacsprocdesc"),
                         ('PACS STUDY DESCRIPTION', "pacsstudydesc"),
                         ('PACS BODY PART', "pacsbodypart"),
                         ('PACS MODALITY', "pacsmodality")]
    x = x[columns]
    x[columns] = x[columns].astype(str)
    tokenizer = RegexpTokenizer(regex_string)
    x['tokens'] =  [[]] * len(x)

    if save_missing_feature_as_string:
        for col in columns_with_desc:
            x[col[0]] = x[col[0]].apply(lambda x: x if x != 'nan' else ("missing" + col[1]))
    else:
        for col in columns_with_desc:
            x[col[0]] = x[col[0]].apply(lambda x: x if x != 'nan' else "")

    for col in columns:
        tokens = x[col].apply(lambda x: ((tokenizer.tokenize(x.lower()))))
        x['tokens'] = x['tokens'] + tokens
    # from IPython import embed
    # embed()
    if remove_repeats:
        x['tokens'] = x['tokens'].apply(lambda y: list(set(y)))
    if remove_short:
        x['tokens'] = x['tokens'].apply(lambda y: [z for z in y if len(z) > 1])
    if remove_num:
        x['tokens'] = x['tokens'].apply(lambda y: [z for z in y if z.isalpha()])
    if remove_empty:
        if (len(x[x['tokens'].str.len() == 0]) != 0):
            y = y[x['tokens'].str.len() != 0]
            x = x[x['tokens'].str.len() != 0]
    return x['tokens'], y

def normalize_east_dir_df(df):
    columns = ['ris_procedure_code', 'ris_procedure_description',
       'pacs_procedure_code', 'pacs_study_description',
       'pacs_procedure_description', 'pacs_body_part', 'pacs_modality']
    new_names =  ['RIS PROCEDURE CODE', 'RIS PROCEDURE DESCRIPTION', 'PACS SITE PROCEDURE CODE', 'PACS PROCEDURE '
                  'DESCRIPTION', 'PACS STUDY DESCRIPTION', 'PACS BODY PART', 'PACS MODALITY']
    df =  df[columns]
    df.columns = new_names
    return df


def get_x_y_split(input_df, label=label_columns):
    df = input_df.drop('src_file', axis=1)

    df_x = df.drop(label, axis=1)
    df_y = df[label]


    return df_x, df_y