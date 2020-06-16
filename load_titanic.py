import numpy as np
import pandas as pd
import os


def age_ranges(data_df_):
    """
    This converts the 'Age' column in titanic data into the median per category
    :param data_df_: titanic dataframe (must include columns 'Age' and 'Pclass'
    :return: modified pandas dataframe
    """
    data_df = data_df_.copy()
    guess_ages = np.zeros((2, 3))

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = data_df[(data_df['Sex'] == i) & (data_df['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            data_df.loc[(data_df.Age.isnull()) & (data_df.Sex == i) & (data_df.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    data_df['Age'] = data_df['Age'].astype(int)
    return data_df


def run_data_preprocess():
    """

    :return: pre-processed data
    """
    # get curr directory
    curr_dir = os.getcwd()
    # import data
    data_train = pd.read_csv(curr_dir + "/data/train.csv")
    data_test = pd.read_csv(curr_dir + "/data/test.csv")

    # split data
    train_y = data_train["Survived"]

    # let's drop the Survived and Name column as they are not needed for our model
    train_x = data_train.drop(['Survived', 'Name'], axis=1)
    test_x = data_test.drop(['Name'], axis=1)

    """DATA EDITS"""
    # UPDATE 'Sex' to a numerical value, male == 1, female == 2
    train_x['Sex'] = np.where(train_x['Sex'] == 'male', 0, 1)
    test_x['Sex'] = np.where(test_x['Sex'] == 'male', 0, 1)

    # 'Embarked' is only missing two values, and we want to convert it to numerical value
    # train set
    train_x['Embarked'] = train_x['Embarked'].fillna(train_x['Embarked'].mode().iloc[0])
    embarked_unique_vals = train_x['Embarked'].unique()
    numbers = list(range(0, len(embarked_unique_vals)))
    dict_embarked = {}
    for idx, embarked_orig_val in enumerate(embarked_unique_vals):
        dict_embarked[embarked_orig_val] = numbers[idx]

    train_x = train_x.replace({"Embarked": dict_embarked})

    # test set
    test_x['Embarked'] = test_x['Embarked'].fillna(test_x['Embarked'].mode().iloc[0])
    embarked_unique_vals = test_x['Embarked'].unique()
    numbers = list(range(0, len(embarked_unique_vals)))
    dict_embarked = {}
    for idx, embarked_orig_val in enumerate(embarked_unique_vals):
        dict_embarked[embarked_orig_val] = numbers[idx]

    test_x = test_x.replace({"Embarked": dict_embarked})

    # CONVERT Age to a categorical values
    modified_age = age_ranges(train_x)
    modified_age_test = age_ranges(test_x)

    train_x['Age'] = modified_age['Age']
    test_x['Age'] = modified_age_test['Age']

    train_x.loc[train_x['Age'] <= 16, 'Age'] = 0
    train_x.loc[(train_x['Age'] > 16) & (train_x['Age'] <= 32), 'Age'] = 1
    train_x.loc[(train_x['Age'] > 32) & (train_x['Age'] <= 48), 'Age'] = 2
    train_x.loc[(train_x['Age'] > 48) & (train_x['Age'] <= 64), 'Age'] = 3
    train_x.loc[train_x['Age'] > 64, 'Age'] = 4

    test_x.loc[train_x['Age'] <= 16, 'Age'] = 0
    test_x.loc[(train_x['Age'] > 16) & (test_x['Age'] <= 32), 'Age'] = 1
    test_x.loc[(train_x['Age'] > 32) & (test_x['Age'] <= 48), 'Age'] = 2
    test_x.loc[(train_x['Age'] > 48) & (test_x['Age'] <= 64), 'Age'] = 3
    test_x.loc[train_x['Age'] > 64, 'Age'] = 4

    # With some of the data cleaned, now drop null columns
    train_x = train_x.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
    test_x = test_x.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)

    return train_x, train_y, test_x