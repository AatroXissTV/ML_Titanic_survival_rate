# exploration_data_analysis.py
# created 24/01/2022 at 15:31 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 17:53 by Antoine 'AatroXiss' BEAUDESSON

""" clean_data.py:

To do:
    - clean the data
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.0.9"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports

# local application imports

# other imports

# constants

# configs


def exploration_data_analysis(cleaner_data, data1_x, target):
    for x in data1_x:
        if cleaner_data[0][x].dtype != 'float64':
            print('Survival Correlation by:', x)
            print(cleaner_data[0][[x, target[0]]].groupby(
                x, as_index=False).mean())
            print('-'*10, '\n')
