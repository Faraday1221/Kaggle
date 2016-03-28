#!/usr/bin/env python
# coding=utf-8
'''
Build our first random forrest with Sci-kit learn
https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests
'''

import pandas as pd
import numpy as np
import os
from pkgs import lookup_table

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# read in the data
path = os.path.dirname(__file__)+'/data/'

filename = 'test.csv'
test = pd.read_csv(os.path.join(path,filename))

filename = 'train.csv'
train = pd.read_csv(os.path.join(path,filename))

# data pre-read
# print train.info()
# print test.info()


#===============================================================================
# Preprocessing
#===============================================================================
# in order to implement the Sci-kit learn random forrest we have to descritize
# the data, there is a lot of clean up to do. see myfirstforest.py for reference


# to fill in AGE (train & test)
# create a look up of median age from Sex, Pclass
group_list = ['Sex','Pclass']
age_dict = train.groupby(group_list).median()['Age'].to_dict()
train.Age.fillna(train.apply(lambda x: lookup_table(x, group_list, age_dict),axis=1),inplace=True)
test.Age.fillna(train.apply(lambda x: lookup_table(x, group_list, age_dict),axis=1),inplace=True)

# descritize AGE
age_strata = [0,14,18,25,30,40,80]
train['AGE'] = pd.cut(train.Age, bins=age_strata, labels=False)
test['AGE'] = pd.cut(test.Age, bins=age_strata, labels=False)

# fill in Fare (test only)
# since there is single missing fare we will be simple about filling the na
test.Fare.fillna(test.Fare.median(),inplace=True)

# descritize Fare (train & test)
train['FARE'], fare_bins = pd.qcut(train.Fare,q=4, precision=0, retbins=True, labels=False)
test['FARE'] = pd.cut(test.Fare, bins=fare_bins, labels=False, include_lowest=True)

# fill in Embarked (train only
# S is most common and we are only missing 2 entries, so here we simply mark "S"
train.Embarked.fillna('S',inplace=True)

# categorize the data (Sex, Embarked)
Embarked_dict = {'C':0,'Q':1,'S':2}
test['EMBARKED'] = test.Embarked.map(Embarked_dict)
train['EMBARKED'] = train.Embarked.map(Embarked_dict)

Sex_dict = {'female':0,'male':1}
test['SEX'] = test.Sex.map(Sex_dict)
train['SEX'] = train.Sex.map(Sex_dict)

#===============================================================================
# Features
#===============================================================================
# we could descritize SibSp & Parch and include it, instead FAMILY is our proxy

# Family size Siblings & Spouses + Parents & Children
# note that we are not looking at the SibSp or Parch individually
train['Family'] = train.SibSp + train.Parch
test['Family'] = test.SibSp + train.Parch

fam_strata = [0,1,2,20]
train['FAMILY'] = pd.cut(train.Family, bins=fam_strata, labels=False, include_lowest=True)
test['FAMILY'] = pd.cut(test.Family, bins=fam_strata, labels=False, include_lowest=True)


# print train.info()
# print test.info()

#===============================================================================
# Reduce to our final Dataset
#===============================================================================
# drop anything we don't need & format as an array
retain = ['Pclass', 'SEX', 'EMBARKED','FARE', 'AGE', 'FAMILY']

TEST = test[retain].values
TRAIN= train[retain].values
CLASSIFIER = train['Survived'].values

#===============================================================================
# Train a Random forest
#===============================================================================

# instantiate the forest
forest = RandomForestClassifier(n_estimators=100)
# train the forest
forest = forest.fit(TRAIN, CLASSIFIER)
# predict the forest
pred = forest.predict(TEST)

# output for submissionshould be PassengerId then pred values
