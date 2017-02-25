"""
the purpose of this notebook is to have a simple repeatable way to measure feature importance as a quick check on any feature engineering. The way to go about this is pretty straight forward, we build models then see which features each model marks as most important. We also need to know how well each model performs to contextualize the feature importance outputs.

the models we will use are:

1. Linear Regression
2. Ridge Regression
3. Decision Tree

this will need to be split between classifiers and regression

what do we need to see as an output?
1. feature importance
    i.  ??
    ii. ??

2. model evaluation
    i.  ??
    ii. ??

"""
# idea for a follow up (this is already kind of started): have a quick stats check and visualization for features as well i.e. description / skew / correlation. essentially we should be able to easily check all the assumptions for a linear model
from __future__ import print_function
import abc

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error


#===============================================================================
# A container for the feature details we want to extract from our models
#===============================================================================


class FeatureScoresRegression(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,X,y,cv=5,scoring='neg_mean_squared_error'):
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.random_state = 0

        self.clf = self._get_clf()
        self.rmse = self._calculate_score()
        self.feature_scores = self._feature_importance()
        self.inlcuded_features = self._include_exclude(include=True)
        self.excluded_features = self._include_exclude(include=False)

    @abc.abstractmethod
    def _get_clf(self):
        pass
        # return LinearRegression().fit(self.X,self.y)

    def _calculate_score(self):
        rmse = np.sqrt(-cross_val_score(self.clf,X=self.X,y=self.y,
                        cv=self.cv,scoring=self.scoring))
        return rmse

    def _feature_importance(self):
        s = pd.DataFrame([self.clf.coef_,self.X.columns]).T
        s.columns = ['coef','feature',]
        return s.sort_values('coef',ascending=False)

    def _include_exclude(self,include=True):
        mask = np.where(np.abs(self.feature_scores.coef) > 1e-4, True,False)
        if include == True:
            #return included features
            return self.feature_scores[mask]
        else:
            return self.feature_scores[mask==False]

    def summary(self,N=10):
        print('='*80)
        # name
        print('Description:\t{0}'.format(self.name))
        # model mean & std rmse
        print('\nRMSE mean and std:\t{0:.3f} +/- {1:.3f}'.format(np.mean(self.rmse) ,np.std(self.rmse)))
        # model parameters
        print(self.clf)
        # number of 0 weighted features
        print('\n{0} features exlcuded'.format(self.excluded_features.shape[0]))
        print('{0} features remaining'.format(self.inlcuded_features.shape[0]))
        # top N features
        print('\nThe top {0} features:\n'.format(N))
        print(self.feature_scores.head(N).reset_index())
        print('='*80)



#===============================================================================
# Individual Model builds that populate the FeatureScoresRegression Class
#===============================================================================


class Linear(FeatureScoresRegression):
    def __init__(self,X,y):
        self.name = 'Linear Regression coefficients.\nBe sure to check for multicolinearity i.e. examine both the head and tail of the coefficients'
        super(Linear,self).__init__(X,y)


class LinearL1(FeatureScoresRegression):
    def __init__(self,X,y):
        self.name = 'Lasso or L1 Linear Regression with light GridSearch alpha.'
        super(LinearL1,self).__init__(X,y)

    def _get_clf(self):
        clf = LassoCV(alphas=[.001,.01,.1,1.,10.,100.],
                        random_state=self.random_state,
                        cv=self.cv).fit(self.X,self.y)
        return clf


class RandomForest(FeatureScoresRegression):
    def __init__(self,X,y):
        self.name = 'RandomForest with light GridSearch for the Forest Params'
        super(RandomForest,self).__init__(X,y)

    def _get_clf(self):
        params = {  'n_estimators': [10,100],
                    'max_depth': [5,20,None],
                    'min_samples_split':[2]}
        clf = RandomForestRegressor(random_state=self.random_state)
        cv = GridSearchCV(estimator=clf,param_grid=params,
                          scoring=self.scoring,cv=self.cv).fit(self.X,self.y)
        return cv.best_estimator_

    def _feature_importance(self):
        s = pd.DataFrame([self.clf.feature_importances_,self.X.columns]).T
        s.columns = ['coef','feature']
        return s.sort_values('coef',ascending=False)

#===============================================================================
# Wrapper to easily run all classes above
#===============================================================================


def feature_importance_regression(X,y,N=10):
    """A wrapper for the three methods of investigating feature importance.

       Expects inputs X and y as dataframes where X is the training data
       and y is the target variable. Optional N as int to show the top N
       features for each model in the summary.

       Trains and Prints the summary information for each of the three models
       returned as the function output.

       Returns the three feature_importance model objects, such that each can
       be investigated in more detail.

          - LinearRegression
          - LassoRegression
          - RandomForest
    """
    a,b,c = Linear(X,y), LinearL1(X,y), RandomForest(X,y)
    for model in [a,b,c]:
        model.summary(N)
    return a,b,c
