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
__version__ = "0.0.7"
__maintainer__ = "Antoine 'AatroXiss' BEAUDESSON"
__email__ = "antoine.beaudesson@gmail.com"
__status__ = "Development"

# standard library imports

# third party imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# local application imports

# other imports

# constants

# configs
TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/gender_submission.csv"


def load_dataset(path):
    """
    Load the data from the given path
    :param path: the path of the data
    :return: a pandas df with the data
    """
    return pd.read_csv(path)


def check_null_value_in_dataset(df, dataset_name):
    """
    Check if there is any null value in the dataset
    :param df: the dataset to check
    :param dataset_name: the name of the dataset
    :return: a list of the columns with null values
    """
    return print(f"{dataset_name} col with null values:\n", df.isnull().sum())


def clean_dataset_fill(data_cleaner):
    """
    Clean the data
    :param data_cleaner: the data to clean
    """
    for dataset in data_cleaner:
        # complete missing age with the median
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        # complete missing embarked with mode
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
        # complete missing fare with median
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)


def clean_dataset_feature_engineering(data_cleaner):
    """
    Check if a passenger is alone with family size
    :param data_cleaner: the data to clean
    """
    for dataset in data_cleaner:
        # discrete variables
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 1  # initialize to yes/1 is alone
        dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
        # split title from name
        dataset['Title'] = dataset['Name'].str.split(
            ", ", expand=True)[1].str.split(
                ".", expand=True)[0]
        # continuous variables bins
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


def clean_dataset(data_cleaner):
    """
    clean the data
    :param data_cleaner: the data to clean
    """
    # datasets
    data_copy = data_cleaner[0]

    # clean the data
    # first let's drop the columns we don't need
    drop_col = ['Cabin', 'PassengerId', 'Ticket']
    data_copy.drop(drop_col, axis=1, inplace=True)
    # now let's clean the data
    clean_dataset_fill(data_cleaner)
    clean_dataset_feature_engineering(data_cleaner)
    # cleanup the rare title names
    stat_min = 10
    title_names = (data_copy['Title'].value_counts() < stat_min)
    # find and replace with fewer lines of code
    data_copy['Title'] = data_copy['Title'].apply(
        lambda x: 'Misc' if title_names.loc[x] == True else x)  # noqa 119


def convert_formats(data_cleaner):

    # datasets
    data_copy = data_cleaner[0]

    # convert categorical data to dummy variables for mathematical analysis
    label = LabelEncoder()
    for dataset in data_cleaner:
        dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
        dataset['Title_Code'] = label.fit_transform(dataset['Title'])
        dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

    # define y variable
    Target = ['Survived']

    # define x variables
    data_copy_x = [
        'Sex',
        'Pclass',
        'Embarked',
        'Title',
        'SibSp',
        'Parch',
        'Age',
        'Fare',
        'FamilySize',
        'IsAlone'
    ]
    data_copy_x_calc = [
        'Sex_Code',
        'Pclass',
        'Embarked_Code',
        'Title_Code',
        'SibSp',
        'Parch',
        'Age',
        'Fare',
    ]
    # coded for algo calc
    data_copy_xy = Target + data_copy_x

    # define x var for original w/bin features to remove continuous variables
    data_copy_x_bin = [
        'Sex_Code',
        'Pclass',
        'Embarked_Code',
        'Title_Code',
        'FamilySize',
        'AgeBin_Code',
        'FareBin_Code'
    ]
    data_copy_xy_bin = Target + data_copy_x_bin

    # define x and y var for dummy features original
    data_copy_dummy = pd.get_dummies(data_copy[data_copy_x])
    data_copy_x_dummy = data_copy_dummy.columns.tolist()
    data_copy_xy_dummy = Target + data_copy_x_dummy

    return Target, data_copy_x_calc, data_copy_x_bin, data_copy_x_dummy, data_copy_dummy, data_copy_x


def split_data(data_cleaner, Target, data_copy_x_calc, data_copy_x_bin, data_copy_x_dummy, data_copy_dummy):
    """
    """
    # datasets
    data_copy = data_cleaner[0]

    # copy
    train_copy_x, test_copy_x, train_copy_y, test_copy_y = train_test_split(
        data_copy[data_copy_x_calc], data_copy[Target], random_state=0)
    # bin
    train_copy_x_bin, test_copy_x_bin, train_copy_y_bin, test_copy_y_bin = train_test_split(
        data_copy[data_copy_x_bin], data_copy[Target], random_state=0)
    # dummy
    train_copy_x_dummy, test_copy_x_dummy, train_copy_y_dummy, test_copy_y_dummy = train_test_split(
        data_copy_dummy[data_copy_x_dummy], data_copy[Target], random_state=0)


def main():
    data_train = load_dataset(TRAIN_PATH)
    data_test = load_dataset(TEST_PATH)

    # To play with our data I will create a copy of the data
    data_copy = data_train.copy(deep=True)

    # Clean both datasets at once
    data_cleaner = [data_copy, data_test]
    clean_dataset(data_cleaner)

    # Convert formats
    Target, data_copy_x_calc, data_copy_x_bin, data_copy_x_dummy, data_copy_dummy, data_copy_x = convert_formats(data_cleaner)

    # Split data into training and testing sets
    split_data(
        data_cleaner,
        Target,
        data_copy_x_calc,
        data_copy_x_bin,
        data_copy_x_dummy,
        data_copy_dummy)

    # Exploratory Data Analysis
    for x in data_copy_x:
        if data_copy[x].dtype != 'float64':
            print('Survival correlation by:', x)
            print(data_copy[[x, Target[0]]].groupby(x, as_index=False).mean())
            print('-' * 20)

    print(pd.crosstab(data_copy['Title'], data_copy[Target[0]]))

    # create graphs
    plt.figure(figsize=(16, 12))

    plt.subplot(231)
    plt.boxplot(x=data_copy['Fare'], showmeans=True, meanline=True)
    plt.title('Fare Boxplot')
    plt.ylabel('Fare ($)')

    plt.subplot(232)
    plt.boxplot(data_copy['Age'], showmeans=True, meanline=True)
    plt.title('Age Boxplot')
    plt.ylabel('Age (years)')

    plt.subplot(233)
    plt.boxplot(data_copy['FamilySize'], showmeans=True, meanline=True)
    plt.title('Family Size Boxplot')
    plt.ylabel('Family Size (#)')

    plt.subplot(234)
    plt.hist(x=[data_copy[data_copy['Survived'] == 1]['Fare'], data_copy[data_copy['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Fare Histogram by Survival')
    plt.xlabel('Fare ($)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(235)
    plt.hist(x=[data_copy[data_copy['Survived'] == 1]['Age'], data_copy[data_copy['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Age Histogram by Survival')
    plt.xlabel('Age (Years)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(236)
    plt.hist(x=[data_copy[data_copy['Survived'] == 1]['FamilySize'], data_copy[data_copy['Survived'] == 0]['FamilySize']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Family Size Histogram by Survival')
    plt.xlabel('Family Size (#)')
    plt.ylabel('# of passengers')
    plt.legend()

    # graph individual features by survival
    fig, saxis = plt.subplots(2, 3, figsize=(16, 12))

    sns.barplot(x='Embarked', y='Survived', data=data_copy, ax=saxis[0, 0])
    sns.barplot(x='Pclass', y='Survived', order=[1, 2, 3], data=data_copy, ax=saxis[0, 1])
    sns.barplot(x='IsAlone', y='Survived', order=[1, 0], data=data_copy, ax=saxis[0, 2])
    sns.pointplot(x='FareBin', y='Survived', data=data_copy, ax=saxis[1, 0])
    sns.pointplot(x='AgeBin', y='Survived', data=data_copy, ax=saxis[1, 1])
    sns.pointplot(x='FamilySize', y='Survived', data=data_copy, ax=saxis[1, 2])


    # graph distribution of qualitative data: Pclass,
    # we know class mattered in survival, so let's graph it
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(14, 12))
    sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data_copy, ax=axis1)
    axis1.set_title('Pclass vs Fare Survival Comparison')

    sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data_copy, split=True, ax=axis2)
    axis2.set_title('Pclass vs Age Survival Comparison')

    sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data_copy, ax=axis3)
    axis3.set_title('Pclass vs Family Size Survival Comparison')


    # graph distribution of qualitative data: Sex
    # we know sex mattered in survival, now let's compare sex and a 2nd feature
    fig, qaxis = plt.subplots(1, 3, figsize=(14, 12))

    sns.barplot(x="Sex", y='Survived', hue="Embarked", data=data_copy, ax=qaxis[0])
    axis1.set_title('Sex vs Embarked Survival comparison')

    sns.barplot(x='Sex', y='Survived', hue='Pclass', data=data_copy, ax = qaxis[1])
    axis1.set_title('Sex vs Pclass Survival comparison')

    sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=data_copy, ax=qaxis[2])
    axis1.set_title('Sex vs IsAlone Survival Comparison')

    # more side-by-sidfe comparisons
    fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=(14, 12))

    # how does family size factor with sex & survival compare
    sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data_copy,
                  palette={"male": "blue", "female": "pink"},
                  markers=["*", "o"],
                  linestyles=["-","--"], ax=maxis1)
    sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_copy,
                palette={"male": "blue", "female": "pink"},
                markers=["*", "o"], linestyles=["-", "--"], ax=maxis2)

    # How does embark port factor with class, sex, and survival
    e = sns.FacetGrid(data_copy, col='Embarked')
    e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep')
    e.add_legend()


    # plot distributions of age of passengers who survived or did not survive
    a = sns.FacetGrid(data_copy, hue='Survived', aspect=4)
    a.map(sns.kdeplot, 'Age', shade=True)
    a.set(xlim=(0, data_copy['Age'].max()))
    a.add_legend()

    # histogram comparision of sexn, class, and age by survival
    h = sns.FacetGrid(data_copy, row='Sex', col='Pclass', hue='Survived')
    h.map(plt.hist, 'Age', alpha=.75)
    h.add_legend()

    # pair plots of entire dataset
    pp = sns.pairplot(data_copy, hue='Survived', palette='deep', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
    pp.set(xticklabels=[])

    correlation_heatmap(data_copy)

    plt.show()



#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

main()
