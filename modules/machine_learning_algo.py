# machine_learning_algo.py
# created 25/01/2022 at 13:39 by Antoine 'AatroXiss' BEAUDESSON
# last modified 25/01/2022 at 16:55 by Antoine 'AatroXiss' BEAUDESSON

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
__version__ = "0.1.4"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# local application imports

# other imports

# constants

# configs


def modeling(dataset):
    y = dataset['Survived']
    x = pd.get_dummies(dataset.drop(['Survived'], axis=1))

    # split model
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_val, y_train, y_val


def fit_and_predict(model, X_train, y_train, X_val, y_val):
    """"
    The following code makes faster to evaluate a model
    automating the fit and accuracy process
    """
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    return accuracy_score(y_val, prediction)


def machine_learning_algorithm(test_copy, train_dataset, test_dataset,
                               x_train, y_train, x_val, y_val):
    """
    Testing different machine learning algorithm
    And getting the accuracy
    """

    model1 = GradientBoostingClassifier()
    model2 = KNeighborsClassifier(3)
    model3 = SVC(probability=True)
    model4 = DecisionTreeClassifier()
    model5 = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        max_features='auto',
        random_state=0,
        oob_score=False,
        min_samples_split=2,
        criterion='gini',
        min_samples_leaf=2,
        bootstrap=False
    )
    model6 = AdaBoostClassifier()
    model7 = GaussianNB()
    model8 = LinearDiscriminantAnalysis()
    model9 = QuadraticDiscriminantAnalysis()

    mla_list = [model1, model2, model3,
                model4, model5, model6,
                model7, model8, model9]

    i = 0
    for model in mla_list:
        i += 1
        print("Model ", i, ":", model)
        print("ACC: ", fit_and_predict(model, x_train, y_train, x_val, y_val))

    predict = model1.predict(pd.get_dummies(test_dataset))
    output = pd.DataFrame({
        'PassengerId': test_dataset.PassengerId,
        'Survived': predict
    })
    output.to_csv('submission.csv', index=False)
    print("Submission saved")
