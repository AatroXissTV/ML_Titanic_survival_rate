# graphs.py
# created 24/01/2022 at 20:10 by Antoine 'AatroXiss' BEAUDESSON
# last modified 24/01/2022 at 20:10 by Antoine 'AatroXiss' BEAUDESSON

""" graphs.py:

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
import matplotlib.pyplot as plt
import seaborn as sns

# local application imports

# other imports

# constants

# configs


def graphs_distribution_of_quantitative_data(cleaned_data):
    plt.figure(figsize=(16, 12))
    plt.subplot(231)
    plt.boxplot(x=cleaned_data[0]['Fare'], showmeans=True, meanline=True)
    plt.title('Fare Boxplot')
    plt.ylabel('Fare ($)')

    plt.subplot(232)
    plt.boxplot(cleaned_data[0]['Age'], showmeans=True, meanline=True)
    plt.title('Age Boxplot')
    plt.ylabel('Age (years)')

    plt.subplot(233)
    plt.boxplot(cleaned_data[0]['FamilySize'], showmeans=True, meanline=True)
    plt.title('Family Size Boxplot')
    plt.ylabel('Family Size (#)')

    plt.subplot(234)
    plt.hist(x=[cleaned_data[0][cleaned_data[0]['Survived'] == 1]['Fare'],
             cleaned_data[0][cleaned_data[0]['Survived'] == 0]['Fare']],
             stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Fare Histogram by Survival')
    plt.xlabel('Fare ($)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(235)
    plt.hist(x=[cleaned_data[0][cleaned_data[0]['Survived'] == 1]['Age'],
             cleaned_data[0][cleaned_data[0]['Survived'] == 0]['Age']],
             stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Age Histogram by Survival')
    plt.xlabel('Age (Years)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(236)
    plt.hist(x=[cleaned_data[0][cleaned_data[0]['Survived'] == 1]['FamilySize'],  # noqa
             cleaned_data[0][cleaned_data[0]['Survived'] == 0]['FamilySize']],
             stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Family Size Histogram by Survival')
    plt.xlabel('Family Size (#)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.show()


def graph_individual_features_by_survival(cleaned_data):
    fig, saxis = plt.subplots(2, 3, figsize=(16, 12))
    sns.barplot(x='Embarked', y='Survived',
                data=cleaned_data[0], ax=saxis[0, 0])
    sns.barplot(x='Pclass', y='Survived',
                order=[1, 2, 3], data=cleaned_data[0], ax=saxis[0, 1])
    sns.barplot(x='IsAlone', y='Survived',
                order=[1, 0], data=cleaned_data[0], ax=saxis[0, 2])
    sns.pointplot(x='FareBin', y='Survived',
                  data=cleaned_data[0], ax=saxis[1, 0])
    sns.pointplot(x='AgeBin', y='Survived',
                  data=cleaned_data[0], ax=saxis[1, 1])
    sns.pointplot(x='FamilySize', y='Survived',
                  data=cleaned_data[0], ax=saxis[1, 2])

    plt.show()


def graph_distribution_of_qualitative_data(cleaned_data):
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(14, 12))

    sns.boxplot(x='Pclass', y='Fare', hue='Survived',
                data=cleaned_data[0], ax=axis1)
    axis1.set_title('Pclass vs Fare Survival Comparison')

    sns.violinplot(x='Pclass', y='Age', hue='Survived',
                   data=cleaned_data[0], split=True, ax=axis2)
    axis2.set_title('Pclass vs Age Survival Comparison')

    sns.boxplot(x='Pclass', y='FamilySize', hue='Survived',
                data=cleaned_data[0], ax=axis3)
    axis3.set_title('Pclass vs Family Size Survival Comparison')

    plt.show()


def pair_plot_dataset(cleaned_data):
    pp = sns.pairplot(cleaned_data[0], hue='Survived',
                      palette='deep', size=1.2, diag_kind='kde',
                      diag_kws=dict(shade=True), plot_kws=dict(s=10))
    pp.set(xticklabels=[])
    plt.show()
