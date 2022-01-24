# main.py
# created 21/01/2022 at 15:09 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 14:43 by Antoine 'AatroXiss' BEAUDESSON

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
__version__ = "0.0.8"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns

# local application imports
from modules.clean_data import clean_data
from modules.load_dataset import load_dataset

# other imports

# constants

# configs
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/gender_submission.csv"


def main():
    # the first part of ML is to load the data
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    # Then I create a copy of the test df and put it in a list
    # To play with the data
    copy_df = train_df.copy(deep=True)
    data_cleaner = [copy_df, test_df]

    # The second part of ML is to clean the data
    clean_data(data_cleaner)

    # The third part of ML is to convert formats


main()
