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
__version__ = "0.1.3"
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

        self.dataset.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
        self.dataset.dropna(axis=0, subset=['Embarked'], inplace=True)

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
