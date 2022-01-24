# clean_data.py
# created 24/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 15:38 by Antoine 'AatroXiss' BEAUDESSON

""" clean_data.py:

To do:
    - clean the data
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.8"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import pandas as pd

# local application imports

# other imports

# constants

# configs


def clean_data(data_cleaner):
    """
    Clear the data in argument
    - First part: drop useless columns
    - Second part: Fill rows with missing values
    - Third part: Feature engineering
    - Fourth part: Clean rare titles found in feature engineering

    :param data_cleaner: The data to clean is composed of a list of df
                         The first df is the train df [0]
                         The second df is the test df [1]
    :return: The cleaned data
    """

    # First part: drop useless columns
    useless_col = ["PassengerId", "Ticket", "Cabin"]
    drop_useless_col(useless_col, data_cleaner[0])

    # Second part: Fill rows with missing values
    fill_dataset(data_cleaner)

    # Third part: Feature engineering
    feature_engineering(data_cleaner)

    # Fourth part: Cleanup rare titles
    stat_min = 10  # we'll use the common minimum in statistics
    title_names = (data_cleaner[0]['Title'].value_counts() < stat_min)
    # find and replace
    data_cleaner[0]['Title'] = data_cleaner[0]['Title'].apply(
        lambda x: 'Misc' if title_names.loc[x] == True else x)  # noqa 119


def drop_useless_col(useless_col, df):
    """
    Drop the useless columns in a df
    given in argument.
    """
    df.drop(useless_col, axis=1, inplace=True)
    return df


def fill_dataset(data_cleaner):
    """
    Fill the dataset with missing values

    :param data_cleaner: The data to clean is composed of a list of df
                         The first df is the train df [0]
                         The second df is the test df [1]
    :return: The cleaned data
    """

    for dataset in data_cleaner:
        # complete missing age with median
        dataset['Age'].fillna(
            dataset['Age'].median(), inplace=True
        )
        # complete embarked with mode
        dataset['Embarked'].fillana(
            dataset['Embarked'].mode()[0], inplace=True
        )
        # complete missing fare with median
        dataset['Fare'].fillna(
            dataset['Fare'].median(), inplace=True
        )


def feature_engineering(data_cleaner):
    """
    Feature engineering is when we use existing features to create new features
    to determine if they provide new signals to predict our outcome.
    Create a title feature to determine if it played a role in survival.
    """
    for dataset in data_cleaner:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 1
        dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
        dataset['Title'] = dataset['Name'].str.split(
            ", ", expand=True)[1].str.split(
                ".", expand=True)[0]
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
