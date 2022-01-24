# split_train_test.py
# created 24/01/2022 at 17:41 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 17:41 by Antoine 'AatroXiss' BEAUDESSON

""" convert_formats.py:

To do:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from sklearn.model_selection import train_test_split

# local application imports

# other imports

# constants

# configs


def split_train_and_test_data(data_cleaner,
                              data1_x_calc,
                              target, data1_x_bin,
                              data_x_dummy,
                              data1_dummy):
    """
    I am splitting the train data and the test data
    with function defaults

    random_state -> seed or control random number generator
    """

    train1_x, test1_x, train1_y, test1_y = train_test_split(
        data_cleaner[0][data1_x_calc],
        data_cleaner[0][target],
        random_state=0)

    train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = train_test_split(
        data_cleaner[0][data1_x_bin],
        data_cleaner[0][target],
        random_state=0)

    train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = train_test_split(  # noqa
        data1_dummy[data_x_dummy],
        data_cleaner[0][target],
        random_state=0)
