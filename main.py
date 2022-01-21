# main.py
# created 21/01/2022 at 15:09 by Antoine 'AatroXiss' BEAUDESSON
# last modified 21/01/2022 at 15:34 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:

To do:
    - Load the data
    - Analyze the data with EDA
    - Clean the data
    - Build a classification model
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import pandas as pd

# third party imports

# local application imports

# other imports

# constants
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/gender_submission.csv"


def load_dataset(path):
    """
    Load the data from the given path
    :param path: the path to the data
    :return: the data
    """
    return pd.read_csv(path)


class ExploratoryDataAnalysis:

    def get_dataset_information(dataset):
        dataset.info()

    def get_per_null_series(dataset):
        per_null_series = dataset.isnull().sum()/len(dataset)*100
        return per_null_series

    def sorted_per_null_series(dataset):
        sorted_per_null_series = dataset.sort_values(ascending=False)
        return sorted_per_null_series

    def examine_null_percentages(self, dataset, dataset_name):
        per_null_series = self.get_per_null_series(dataset)
        sorted_per_null_series = self.sorted_per_null_series(per_null_series)
        temp = pd.DataFrame({"Missing Ratio in {}".format(dataset_name): sorted_per_null_series})  # noqa
        return temp


def main():
    # load the data
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    # Examine Null Percentages in train attributwise
    temp_train_df = ExploratoryDataAnalysis.examine_null_percentages(train_df,
                                                                     "train")
    temp_train_df.head()

    # Examine Null Percentages in test attributwise
    temp_test_df = ExploratoryDataAnalysis.examine_null_percentages(test_df,
                                                                    "test")
    temp_test_df.head()


main()
