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
from sklearn.cross_validation import KFold

def load_titanic():
    '''THIS NEEDS A FIX TO THE ABS PATH... I dont love the hard-coding'''
    # read in the data
    # path = os.path.dirname(__file__)+'/data/'

    path = './data/'
    filename = 'test.csv'
    test = pd.read_csv(os.path.join(path,filename))

    filename = 'train.csv'
    train = pd.read_csv(os.path.join(path,filename))
    return train, test

# data pre-read
# print train.info()
# print test.info()


#===============================================================================
# Preprocessing
#===============================================================================
# in order to implement the Sci-kit learn random forrest we have to descritize
# the data, there is a lot of clean up to do. see myfirstforest.py for reference

def fill_in_titanic(train,test):
    # to fill in AGE (train & test)
    # create a look up of median age from Sex, Pclass
    group_list = ['Sex','Pclass']
    age_dict = train.groupby(group_list).median()['Age'].to_dict()
    train.Age.fillna(train.apply(lambda x: lookup_table(x, group_list, age_dict),axis=1),inplace=True)
    test.Age.fillna(train.apply(lambda x: lookup_table(x, group_list, age_dict),axis=1),inplace=True)

    # fill in Fare (test only)
    # since there is single missing fare we will be simple about filling the na
    test.Fare.fillna(test.Fare.median(),inplace=True)

    # fill in Embarked (train only
    # S is most common and we are only missing 2 entries, so here we simply mark "S"
    train.Embarked.fillna('S',inplace=True)
    return train, test

def categorize_titanic(train,test):
    # categorize the data (Sex, Embarked)
    Embarked_dict = {'C':0,'Q':1,'S':2}
    test['EMBARKED'] = test.Embarked.map(Embarked_dict)
    train['EMBARKED'] = train.Embarked.map(Embarked_dict)

    Sex_dict = {'female':0,'male':1}
    test['SEX'] = test.Sex.map(Sex_dict)
    train['SEX'] = train.Sex.map(Sex_dict)
    return train, test

def descritize_titanic(train, test):
    # descritize Fare (train & test)
    train['FARE'], fare_bins = pd.qcut(train.Fare,q=4, precision=0, retbins=True, labels=False)
    test['FARE'] = pd.cut(test.Fare, bins=fare_bins, labels=False, include_lowest=True)

    # descritize AGE
    age_strata = [0,14,18,25,30,40,80]
    train['AGE'] = pd.cut(train.Age, bins=age_strata, labels=False)
    test['AGE'] = pd.cut(test.Age, bins=age_strata, labels=False)
    return train, test

def process_all_titanic(train, test):
    a,b = fill_in_titanic(train,test)
    a,b = categorize_titanic(a,b)
    a,b = descritize_titanic(a,b)
    return a, b
#===============================================================================
# Features
#===============================================================================
# we could descritize SibSp & Parch and include it, instead FAMILY is our proxy

def family_features(train, test):
    # Family size Siblings & Spouses + Parents & Children
    # note that we are not looking at the SibSp or Parch individually
    train['Family'] = train.SibSp + train.Parch
    test['Family'] = test.SibSp + train.Parch

    fam_strata = [0,1,2,20]
    train['FAMILY'] = pd.cut(train.Family, bins=fam_strata, labels=False, include_lowest=True)
    test['FAMILY'] = pd.cut(test.Family, bins=fam_strata, labels=False, include_lowest=True)

    train['MULT_TOP'] = train.Pclass * train.Age * train.SibSp
    test['MULT_TOP'] = test.Pclass * test.Age * test.SibSp

    train['SUM_TOP'] = train.Pclass + train.Age + train.SibSp
    test['SUM_TOP'] = test.Pclass + test.Age + test.SibSp
    return train, test


#===============================================================================
# Helper Functions - modified from Kushal Agarwal
#===============================================================================
# https://www.kaggle.com/kushal1412/titanic/random-forest-survivors

'''it would be great to better vizualize these scores, at the default values of
   range depth (1:100) and estimators (1:300) i suspect we are crazy overfit'''

def best_max_depth(train_data, predictors, target_col):
    max_score = 0
    best_m = 0
    for m in range(1,15):
        rfc_scr = 0.
        rfc = RandomForestClassifier(max_depth=m)

        for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
            X = train_data[predictors].T[train].T
            y = train_data[target_col].T[train].T
            rfc.fit(X,y)
            rfc_scr += rfc.score(X,y)/10
        if rfc_scr > max_score:
            max_score = rfc_scr
            best_m = m

    print 'The best mean accuracy is {1:.2f}%  for max_depth = {0}'.format(best_m, max_score)
    return best_m

def best_n_estimators(train_data, predictors, target_col):
    max_score = 0
    best_n = 0
    for n in range(1,30):
        rfc_scr = 0.
        rfc = RandomForestClassifier(n_estimators=n)

        for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
            X = train_data[predictors].T[train].T
            y = train_data[target_col].T[train].T
            rfc.fit(X,y)
            rfc_scr += rfc.score(X,y)/10
        if rfc_scr > max_score:
            max_score = rfc_scr
            best_n = n
    print 'The best mean accuracy is {1:.2f}%  for n_estimators = {0}'.format(best_n, max_score)
    return best_n

def create_submission(rfc, train, test, predictors, filename):
    rfc.fit(train[predictors], train["Survived"])
    predictions = rfc.predict(test[predictors])
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)


#===============================================================================
# Train a Random forest
#===============================================================================
if __name__ == "__main__":
    # load the data
    train, test = load_titanic()

    # Preprocessing & feature engineering
    train, test = process_all_titanic(train, test)
    train, test = family_features(train, test)

    # drop anything we don't need & format as an array
    retain = ['Pclass', 'SEX', 'EMBARKED','FARE', 'AGE', 'FAMILY']

    print 'Finding the best max_depth...'
    m = best_max_depth(train, retain, 'Survived') #69
    print 'Finding the best n_estimators...'
    n = best_n_estimators(train, retain, 'Survived') # 245

    rfc = RandomForestClassifier(max_depth=m, n_estimators=n)
    print 'Creating submission...'
    create_submission(rfc, train, test, retain, 'first_forest.csv')
    print 'Complete.'


    '''it would be fantastic to find a better way to evaluate our model'''


#===============================================================================
# The first attempt at a Random Forest
#===============================================================================

# TEST = test[retain].values
# TRAIN= train[retain].values
# CLASSIFIER = train['Survived'].values
#
# # instantiate the forest
# forest = RandomForestClassifier(n_estimators=100)
# # train the forest
# forest = forest.fit(TRAIN, CLASSIFIER)
# # predict the forest
# pred = forest.predict(TEST)
#
# # output for submissionshould be PassengerId then pred values
