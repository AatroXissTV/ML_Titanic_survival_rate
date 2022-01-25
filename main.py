# main.py
# created 21/01/2022 at 15:09 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 15:16 by Antoine 'AatroXiss' BEAUDESSON

""" main.py:

To do:
    - Refactor the code to make it more readable.
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

# third party imports

# local application imports
from modules.data_management import LoadDataset, CleanData
from modules.machine_learning_algo import (
    modeling,
    machine_learning_algorithm,
)

# other imports

# constants

# configs
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/gender_submission.csv"


def main():

    train = LoadDataset(TRAIN_PATH)
    train_dataset = train.load()
    cleaning_data = CleanData(train_dataset)
    cleaning_data.drop_useless_info()
    cleaning_data.fill_missing_values_age()
    cleaning_data.fill_missing_values_fare()
    cleaning_data.fill_missing_values_embarked()
    cleaning_data.replace_sex_values_by_numeric()
    cleaning_data.replace_embarked_values_by_numeric()
    cleaning_data.create_new_feature_family_size()
    cleaning_data.create_new_feature_is_alone()
    cleaning_data.create_new_feature_title()
    cleaning_data.categorize_titles()
    cleaning_data.replace_title_values_by_numeric()
    cleaning_data.feature_selection()

    test = LoadDataset(TEST_PATH)
    test_dataset = test.load()
    cleaning_data = CleanData(test_dataset)
    cleaning_data.drop_useless_info()
    cleaning_data.fill_missing_values_age()
    cleaning_data.fill_missing_values_fare()
    cleaning_data.fill_missing_values_embarked()
    cleaning_data.replace_sex_values_by_numeric()
    cleaning_data.replace_embarked_values_by_numeric()
    cleaning_data.create_new_feature_family_size()
    cleaning_data.create_new_feature_is_alone()
    cleaning_data.create_new_feature_title()
    cleaning_data.categorize_titles()
    cleaning_data.replace_title_values_by_numeric()

    # copy train
    test_copy = test_dataset.copy()
    cleaning_data.feature_selection()

    x_train, x_val, y_train, y_val = modeling(train_dataset)
    machine_learning_algorithm(test_copy, train_dataset,
                               test_dataset, x_train,
                               y_train, x_val, y_val)


main()
