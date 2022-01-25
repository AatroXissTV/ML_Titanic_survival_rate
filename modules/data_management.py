# data_management.py
# created 25/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON
# last modified 25/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON

""" data_management.py:

To do:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.4"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import re

# third party imports
import pandas as pd

# local application imports

# other imports

# constants

# configs


class LoadDataset:

    def __init__(self, path):
        self.path = path

    def load(self):
        """
        Load the dataset from the path given in argument.

        :param path: The path of the dataset to load.
        :return: The dataset loaded.
        """
        return pd.read_csv(self.path)


class CleanData:

    def __init__(self, dataset):
        self.dataset = dataset

    def drop_useless_info(self):
        """
        Drop the useless information in the
        dataset given in argument.

        :param dataset: The dataset to clean.
        :return: The cleaned dataset.
        """

        self.dataset.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

    def fill_missing_values_age(self):
        """
        Filling the missing age values
        with the median value

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['Age'] = d.groupby(['Pclass', 'Sex'])['Age'].transform(
            lambda x: x.fillna(x.median()))

    def fill_missing_values_fare(self):
        """
        Filling the missing fare values
        with the median value

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['Fare'] = d.groupby(['Pclass', 'Sex'])['Fare'].transform(
            lambda x: x.fillna(x.median()))

    def fill_missing_values_embarked(self):
        """
        Filling the missing embarked values
        with the most frequent value

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['Embarked'] = d['Embarked'].fillna('S')

    def replace_sex_values_by_numeric(self):
        """
        Replacing sex values in the dataset
        by numeric values.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """

        self.dataset['Sex'].replace(
            {
                'male': 0,
                'female': 1,
            },
            inplace=True
        )

    def replace_embarked_values_by_numeric(self):
        """
        Replacing embarked values in the dataset
        by numeric values

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """

        self.dataset['Embarked'].replace(
            {
                'S': 0,
                'C': 1,
                'Q': 2,
            },
            inplace=True
        )

    def replace_title_values_by_numeric(self):
        """
        Replace the title values by numeric values.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['Title'] = d['Title'].replace(
            ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'],
            [0, 1, 2, 3, 4])

    def create_new_feature_family_size(self):
        """
        Create a new feature called family size
        which is the sum of the number of
        passengers in the family.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

    def create_new_feature_is_alone(self):
        """
        Create a new feature called is alone
        which is 1 if the passenger is alone
        and 0 if he is not.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['IsAlone'] = 0
        d.loc[d['FamilySize'] == 1, 'IsAlone'] = 1

    def create_new_feature_title(self):
        """
        Create a new feature called title
        which is the title of the passenger.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        d['Title'] = d['Name'].apply(get_title)

    def categorize_titles(self):
        """
        Categorize the titles of the passengers
        in the dataset.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """

        d = self.dataset
        d['Title'] = d['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col',
             'Don', 'Dr', 'Major', 'Rev', 'Sir',
             'Jonkheer', 'Dona'], 'Rare')

        d['Title'] = d['Title'].replace('Mlle', 'Miss')
        d['Title'] = d['Title'].replace('Ms', 'Miss')
        d['Title'] = d['Title'].replace('Mme', 'Mrs')

    def feature_selection(self):
        """
        Feature selection of the dataset.

        :param dataset: The dataset to clean.
        :return: the cleaned dataset
        """
        d = self.dataset
        drop_elements = ['Name', 'SibSp', 'Parch', 'FamilySize']
        d.drop(drop_elements, axis=1, inplace=True)


def get_title(name):
    """
     Get the title of the passenger.
    """

    title_search = re.search(' ([A-Za-z]+)\.', name)  # noqa
    if title_search:
        return title_search.group(1)
    return ""
