# main.py
# created 21/01/2022 at 15:09 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 11:04 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:

To do:
    - Analyze the data with EDA
    - Clean the data
    - Build a classification model
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.4"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# third party imports

# local application imports

# other imports

# constants
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/gender_submission.csv"

# configs
sns.set(style='white', context='notebook', palette='deep')


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
    eda = ExploratoryDataAnalysis
    temp_tdf = ExploratoryDataAnalysis.examine_null_percentages(eda,
                                                                train_df,
                                                                "train")
    print(temp_tdf.head())

    # Examine Null Percentages in test attributwise
    temp_tdf = ExploratoryDataAnalysis.examine_null_percentages(eda,
                                                                test_df,
                                                                "test")
    print(temp_tdf.head())

    # Display percentage of survivors per class
    sns.histplot(x="Survived", stat="percent", data=train_df)
    per_class_0 = round(((train_df.loc[:, "Survived"] == 0).sum() / len(train_df)) * 100, 2)  # noqa
    per_class_1 = round(((train_df.loc[:, "Survived"] == 1).sum() / len(train_df)) * 100, 2)  # noqa

    for i, fr in [(0, per_class_0), (0.92, per_class_1)]:
        plt.text(i, fr+0.1, str(fr))

    plt.show()

    # Examine percentage of male and female survivd
    per_males = (
        (train_df.loc[:, "Sex"] == "male") & (
            train_df.loc[:, "Survived"] == 1)).sum()/(
                train_df.loc[:, "Sex"] == "male").sum()
    per_females = (
        (train_df.loc[:, "Sex"] == "female") & (
            train_df.loc[:, "Survived"] == 1)).sum()/(
                train_df.loc[:, "Sex"] == "female").sum()

    plt.bar(x=["male", "female"], height=[per_males, per_females])
    plt.ylabel("Survived(%)")

    for x, per in [(0, per_males), (0.92, per_females)]:
        plt.text(x, per+0.005, str(round(per, 2)))
    plt.show()


main()
