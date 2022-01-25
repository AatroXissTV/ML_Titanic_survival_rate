# mla.py
# created 25/01/2022 at 13:39 by Antoine 'AatroXiss' BEAUDESSON
# last modified 25/01/2022 at 13:39 by Antoine 'AatroXiss' BEAUDESSON

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
__version__ = "0.1.1"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
from sklearn import (
    ensemble,
    model_selection,
)
from sklearn import svm
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# local application imports

# other imports

# constants

# configs


def choose_mla(cleaned_data, target, data1_x_bin):

    mla = [

        # Ensemble Methods: Bagging, Boosting, Tree
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),

        # XGBC
        XGBClassifier()
    ]

    # split dataset in cross validation with this splitter class
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3,
                                            train_size=.6, random_state=0)

    # Create table to compare MLA Metrics
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean',
                   'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD',
                   'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # create table to compare MLA predictions
    MLA_predict = cleaned_data[0][target]

    # index through MLA and save performance to table
    row_index = 0
    print(row_index)
    for alg in mla:

        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        # score model with cross validation
        cv_results = model_selection.cross_validate(
            alg,
            cleaned_data[0][data1_x_bin],
            cleaned_data[0][target],
            cv=cv_split,
            return_train_score=True)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()  # noqa
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean() # noqa

        # if this a non bias random sample, then +/- 3 standard deviations
        # std) from the mean,
        # should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3  # noqa

        # save MLA predictions
        alg.fit(cleaned_data[0][data1_x_bin], cleaned_data[0][target])
        MLA_predict[MLA_name] = alg.predict(cleaned_data[0][data1_x_bin])

        row_index += 1

    # print and sort table
    MLA_compare.sort_values(by='MLA Test Accuracy Mean',
                            ascending=False, inplace=True)

    sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name',
                data=MLA_compare, color='m')
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algo')
    plt.show()


def RandomForestClassifier(cleaned_data, target):
    y = cleaned_data[0][target]
    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    X = pd.get_dummies(cleaned_data[0][features])
    X_test = pd.get_dummies(cleaned_data[1][features])

    model = ensemble.RandomForestClassifier(
        max_depth=3,
        n_estimators=100,
        random_state=2,
    )
    model.fit(X, y)
    model.predict(X_test)

    submission = pd.read_csv('dataset/gender_submission.csv')
    submission.to_csv('submission.csv', index=False)
