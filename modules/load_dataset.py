# load_dataset.py
# created 24/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON

""" load_dataset.py:

To do:
    - load the data
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
import pandas as pd

# local application imports

# other imports

# constants

# configs


def load_dataset(path):
    """
    Load the dataset from the path given in argument.

    :param path: The path of the dataset to load.
    :return: The dataset loaded.
    """
    return pd.read_csv(path)
