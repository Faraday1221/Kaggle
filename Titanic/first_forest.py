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
import re
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

    title_dict={}
    for i, nm in enumerate(np.unique(train.title)):
        title_dict[nm]=i

    train['TITLE'] = train.title.map(title_dict)
    test['TITLE'] = test.title.map(title_dict)
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


def extract_title(series):
    '''exploit the pattern, last name "," then title "."  '''
    title = series.apply(lambda x: re.search(r'[a-zA-Z]+\.',x).group())
    return title

def extract_last(series):
    '''exploit the pattern, last name "," then title "."  '''
    last = series.apply(lambda x: x.split(',')[0].strip())
    return last

def process_title(series,n):
    '''return only the top titles in the dataset, convert anything lower than
       counts > n to rare, as an array'''
    title_counts = series.value_counts()
    title_list = list(title_counts[title_counts > n].index)
    title_array = np.where(series.isin(title_list),series,'rare')
    return title_array

def process_all_titanic(train, test):
    a,b = fill_in_titanic(train,test)

    # set up new features based on the Name field
    a['title'] = process_title(extract_title(a.Name),5)
    b['title'] = process_title(extract_title(b.Name),5)
    a['last'] = extract_last(a.Name)
    b['last'] = extract_last(b.Name)

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

    fam_strata = [0,1,2,5,20]
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

    #===========================================================================
    # split train into a train & test set, tune & compare models
    #===========================================================================
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    retain = ['Pclass','SibSp','Parch','Age',
              'SEX', 'EMBARKED','FARE', 'AGE', 'FAMILY','TITLE',
              'MULT_TOP','SUM_TOP']

    x_train,x_test,y_train,y_test = train_test_split(
                            train[retain].values,
                            train['Survived'].values,
                            test_size=0.3,
                            random_state=0)

    #===========================================================================
    # Logistic Regression
    #===========================================================================
    print 'Logistic Regression'
    pipe_lr = Pipeline([('scl',MinMaxScaler()),
                        ('clf',LogisticRegression(random_state=0))])
    pipe_lr.fit(x_train,y_train)
    print 'Test Accuracy {0:.3f}'.format(pipe_lr.score(x_test,y_test))

    scores=cross_val_score(estimator=pipe_lr,X=x_train,y=y_train,cv=10,n_jobs=1)
    print 'CV Accuracy {0:.3f} +/- {1:.3f}'.format(np.mean(scores),np.std(scores))

    #===========================================================================
    # Random Forrest
    #===========================================================================
    '''this appears to have the worst performance of the three models'''
    print '\nRandom Forest'
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x_train,y_train)
    print 'Test Accuracy {0:.3f}'.format(forest.score(x_test,y_test))
    scores = cross_val_score(estimator=forest,X=x_train,y=y_train,cv=10,n_jobs=1)
    print 'CV Accuracy {0:.3f} +/- {1:.3f}'.format(np.mean(scores),np.std(scores))

    #===========================================================================
    # KNN
    #===========================================================================
    # this performs well on the training set, but does not generalize as well
    # as SVM, thanks cross validation!
    print '\nKNN'
    pipe_kn = Pipeline([('scl',StandardScaler()),
                         ('clf',KNeighborsClassifier())])
    pipe_kn.fit(x_train,y_train)
    print 'Test Accuracy {0:.3f}'.format(pipe_kn.score(x_test,y_test))

    scores=cross_val_score(estimator=pipe_kn,X=x_train,y=y_train,cv=10,n_jobs=1)
    print 'CV Accuracy {0:.3f} +/- {1:.3f}'.format(np.mean(scores),np.std(scores))

    #===========================================================================
    # SVM
    #===========================================================================
    print '\nSVM'
    pipe_svm = Pipeline([('scl',StandardScaler()),
                         ('clf',SVC(random_state=0))])
    pipe_svm.fit(x_train,y_train)
    print 'Test Accuracy {0:.3f}'.format(pipe_svm.score(x_test,y_test))

    scores=cross_val_score(estimator=pipe_svm,X=x_train,y=y_train,cv=10,n_jobs=1)
    print 'CV Accuracy {0:.3f} +/- {1:.3f}'.format(np.mean(scores),np.std(scores))

    # create submission
    create_submission(pipe_svm,train,test,retain,'default_svm.csv')

    #===========================================================================
    # Sequential Feature Selection
    #===========================================================================
    '''THIS IS STILL BROKEN...'''
    # import pkgs
    # pipe_svm = Pipeline([('scl',StandardScaler()),
    #                      ('clf',SVC(random_state=0))])
    # print '\nRunning SBS...'
    # sbs = pkgs.SBS(pipe_svm, k_features=1)
    # sbs.fit(x_train, y_train)
    # print 'Complete.'
    #
    # print 'Lets try the plot...'
    # # plot the Accuracy vs Number of Features
    # import matplotlib.pyplot as plt
    # k_feat = [len(k) for k in sbs.subsets_]
    # plt.plot(k_feat, sbs.scores_,marker='o')
    # # plt.ylim
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of features')
    # plt.grid()
    # plt.show()

    #===========================================================================
    # Reviewing the SVM Validation Curve (check overfitting)
    #===========================================================================
    from sklearn.learning_curve import validation_curve
    import matplotlib.pyplot as plt
    param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0]

    train_scores,test_scores = validation_curve(
        estimator = pipe_svm, X=x_train, y=y_train, param_name='clf__C',
        param_range=param_range, cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5,
             label='training_accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean-train_std,
             alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', marker='s', markersize=5,
             linestyle='--', label='training_accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std,
             alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    # plt.ylim([0.8, 1.0])
    plt.show()

    #===========================================================================
    # Checking Parameter Importance
    #===========================================================================
    '''it would be good to try SBS here if we can use it with SVM (else KNN p.121)'''
    '''otherwise the RandomForest is a way to check importance (p.125)'''
    '''it seems strange to me that we have to use other models to investigate
       how important a feature is'''

    #===========================================================================
    # Tuning our model with GridSearch (p.186)
    #===========================================================================
    '''commenting this out because it is a resource suck'''
    # note we are using the pipe_svm and param_range from the proceeding code
    # spoiler alert: since our default SVM performed so well, we do not see
    # much deviation from our default values, nor improved score
    # in fact, they both make exactly the same predictions!

    # from sklearn.grid_search import GridSearchCV
    #
    # print '\nTuning the SVM model with Grid Search'
    # param_grid = [{'clf__C': param_range,
    #                 'clf__kernel':['linear']},
    #               {'clf__C': param_range,
    #                 'clf__gamma': param_range,
    #                 'clf__kernel': ['rbf']}]
    # gs = GridSearchCV(estimator=pipe_svm, param_grid=param_grid,
    #                     scoring='accuracy', cv=10, n_jobs=-1)
    # gs = gs.fit(x_train,y_train)
    # print gs.best_score_
    # print gs.best_params_
    #
    # # we can use the suggested parameters above to implement the tuned model
    # clf = gs.best_estimator_
    # clf.fit(x_train,y_train)
    # print 'Test Accuracy {0:.3f}'.format(clf.score(x_test,y_test))

    # create submission
    # create_submission(clf,train,test,retain,'gridsearch_svm.csv')




   #===========================================================================
   # Old Code
   #===========================================================================

    # drop anything we don't need & format as an array
    # retain = ['Pclass', 'SEX', 'EMBARKED','FARE', 'AGE', 'FAMILY','TITLE']

    # print 'Finding the best max_depth...'
    # m = best_max_depth(train, retain, 'Survived') #69
    # print 'Finding the best n_estimators...'
    # n = best_n_estimators(train, retain, 'Survived') # 245
    #
    # rfc = RandomForestClassifier(max_depth=m, n_estimators=n)
    # print 'Creating submission...'
    # create_submission(rfc, train, test, retain, 'first_forest.csv')
    # print 'Complete.'


    '''it would be fantastic to find a better way to evaluate our model...
       this is addressed above :) '''


    #===========================================================================
    # The first attempt at a Random Forest
    #===========================================================================

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
