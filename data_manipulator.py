from sklearn.model_selection import train_test_split


def get_train_test_split(self, random_state=1337, label='ON WG IDENTIFIER', test_size=0.25):
    # src_file is not a feature
    df = self.df.drop('src_file', axis=1)

    train, test = train_test_split(df, random_state=random_state, test_size=test_size, shuffle=True)

    train_x = train.drop(label, axis=1)
    train_y = train[label]
    test_x = test.drop(label, axis=1)
    test_y = test[label]

    return train_x, train_y, test_x, test_y


def get_train_validate_test_split(self, random_state=1337, label='ON WG IDENTIFIER', test_size=0.125,
                                  validation_split=0.125):
    # src_file is not a feature
    df = self.df.drop('src_file', axis=1)

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