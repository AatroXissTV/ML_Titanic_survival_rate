# Titanic - Machine Learning from Disaster

## The challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history. 

On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

While there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others.

In this challenge, I am asked to build a predictive model to determine how likely a passenger survived the sinking given some passenger data.

This challenge can be found in the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

### Goal

It's my job to predict whether a passenger survived the sinking or not. For each in the test, I must predict a 0 or 1 value for the variable.

### Data

The data has been split into two files:
* train.csv - contains the data used to train the model
* test.csv - contains the data that will be used to test the model

## Exploratory Data Analysis Summary

When loading the train dataset and looking at the data, I noticed the followings: 
* 77% of missing values for 'Cabin' (687/891) -> Drop
* 20% of missing valuses for 'Age' (177/891) -> Fill
* 0.2% of missing values for 'Embarked' (2/891) -> Fill

We also have categorical variables that need to be taken into account or dropped.
* 'Sex' is a very usefull information. It has 2 possible values
* 'Name' could be a usefull information but for the sake of keeping it simple I will drope this column
* 'Ticket' doesn't give a usefull information to answer to our problem so this column will be dropped too
* 'Cabin' could have been a very usefull information but the dataset has too many missing values so filling is not a good idea
* 'Embarked' is a usefull value to our problem.