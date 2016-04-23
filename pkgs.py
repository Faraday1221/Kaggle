#!/usr/bin/env python
# coding=utf-8

# lookup table helper
def lookup_table(df,col_list, lookup_dict):
    '''expects a dataframe: df
       a list of columns in the data frame: col_list
       and a lookup dictionary: lookup_dict
       NOTE: col_list must be in the same order as the index of lookup_dict

       # Examples:
       # Retrieving Fare values based on Sex and Pclass lookup
       test.apply(lambda x: lookup_table(x,['Sex','Pclass'],fare_dict),axis=1)

       # Retrieving Fare values based on Position
       for i in range(test.shape[0]):
           lookup_table(test.iloc[i],['Sex','Pclass'],fare_dict)

       # Filling in NA values with the lookup table
       test.Fare.fillna(test.apply(lambda x:
            lookup_table(x,['Sex','Pclass'],fare_dict),axis=1), inplace=True)
       '''
    x = tuple(df[i] for i in col_list)
    return lookup_dict[x]

# # EXAMPLE OF THIS SCRIPT IN USE
# %run ./Titanc/first_steps.py
# # we can build a lookup dict fairly simply as follows:
# fare_table = test.groupby(['Sex','Pclass']).mean()['Fare']
# fare_dict = fare_table.to_dict()
# # ultimatley we can fill our null values like below using our lookup table
# test.Fare.fillna(test.apply(lambda x: lookup_table(x,['Sex','Pclass'],fare_dict)
# ,axis=1), inplace=True)


#===============================================================================
# Sequential Feature Selection
#===============================================================================
# python machine learning ch 4 p. 119
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
        test_size=0.25, random_state=0):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self,X,y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X,y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]
        self.indicies_ = tuple(range(dim))
        self.subsets_ = [self.indicies_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indicies_)

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indicies_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indicies_ = subsets[best]
            self.subsets_.append(self.indicies_)
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indicies_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indicies):
        self.estimator.fit(X_train[:,indicies], y_train)
        y_pred = self.estimator.predict(X_test[:, indicies])
        score = self.scoring(y_test, y_pred)
        return score
