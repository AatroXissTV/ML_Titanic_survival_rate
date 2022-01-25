# convert_formats.py
# created 24/01/2022 at 16:18 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 16:18 by Antoine 'AatroXiss' BEAUDESSON

""" convert_formats.py:

To do:
    - *
"""

__author__ = "Antoine 'AatroXiss' BEAUDESSON"
__copyright__ = "Copyright 2021, Antoine 'AatroXiss' BEAUDESSON"
__credits__ = ["Antoine 'AatroXiss' BEAUDESSON"]
__license__ = ""
__version__ = "0.1.1"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# local application imports

# other imports

# constants

# configs


def convert_formats(data_cleaner):
    """
    I am converting catégorical data to dummy var
    categorical data to dummy variables for mathematical analysis.
    There are multiple ways to encode categorical variables;
    I will use the sklearn and pandas functions.
    """

    label = LabelEncoder()

    for dataset in data_cleaner:
        dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
        dataset['Title_Code'] = label.fit_transform(dataset['Title'])
        dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    # define the target var (y)
    Target = ["Survived"]  # noqa

    # define x var for original features aka feature selection
    data1_x = [
        'Sex',
        'Pclass',
        'Embarked',
        'Title',
        'SibSp',
        'Parch',
        'Age',
        'Fare',
        'FamilySize',
        'IsAlone',
    ]  # pretty name/values for charts

    data1_x_calc = [
        'Sex_Code',
        'Pclass',
        'Embarked_Code',
        'Title_Code',
        'SibSp',
        'Parch',
        'Age',
        'Fare'
    ]  # coded for algo calculation

    # define x var for original w/bin features to remove continuous variables
    data1_x_bin = [
        'Sex_Code',
        'Pclass',
        'Embarked_Code',
        'Title_Code',
        'FamilySize',
        'AgeBin_Code',
        'FareBin_Code'
    ]

    # defin x and y var for dummy features original
    data1_dummy = pd.get_dummies(data_cleaner[0][data1_x])
    data_x_dummy = data1_dummy.columns.tolist()

    return Target, data1_x, data1_x_calc, data1_x_bin, data_x_dummy, data1_dummy  # noqa
