from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

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

def remove_stop_words(tokens):
    stopset = set(stopwords.words('english'))
    return [w for w in tokens if (not w in stopset or w != 'nan')]

# Takes as input features as x, and labels as y.
# Returns a pandas series containing the all features tokenized in a list for each row
# Tokenizer considers alphanumeric characters only (can specify using parameters)
# Optional parameter save_missing_as_feature sets all missing columns to a unique
# code to save that that column was missing. Default setting does not do that.
def tokenize(x, y, save_missing_feature_as_string=False, regex_string=r'[a-zA-Z0-9]+'):
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
        tokens = x[col].apply(lambda x: (remove_stop_words(tokenizer.tokenize(x.lower()))))
        x['tokens'] = x['tokens'] + tokens

    return x['tokens'], y