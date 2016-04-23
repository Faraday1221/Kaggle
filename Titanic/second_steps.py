#!/usr/bin/env python
# coding=utf-8

'''
the first_steps model out-performs our first forest RandomForestClassifier,
this is an attempt to beat the first_steps benchmark. The plan is as follows:

1. Evaluate which parameters are most significant
    It would be good to know how well we can do with the parameters prior to
    feature engineering

'''

import numpy as np
import pandas as pd

# we functionalized our first_forest file to allow us to easily Preprocessing
from first_forest import load_titanic, fill_in_titanic, categorize_titanic

train, test = load_titanic()
train, test = fill_in_titanic(train,test)
# tain, test = categorize_titanic(train,test)

# dropping some fields
def drop_list(df, col_to_drop):
    return df.drop(col_to_drop, axis=1)

train = drop_list(train, ['PassengerId','Name', 'Ticket','Cabin'])
test = drop_list(test, ['Name', 'Ticket','Cabin'])

#===============================================================================
# sklearn categorize
#===============================================================================
# we can categorize Sex and Embarked
'''note that the inverse 'map' is overwritten when we use class_le multiple
times, we could probably improve on this implementation... but not now'''

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

train['Sex'] = class_le.fit_transform(train['Sex'].values)
test['Sex'] = class_le.fit_transform(test['Sex'].values)
print np.unique(class_le.inverse_transform(train['Sex']))

train['Embarked'] = class_le.fit_transform(train['Embarked'].values)
test['Embarked'] = class_le.fit_transform(test['Embarked'].values)
print np.unique(class_le.inverse_transform(train['Embarked']))

target = train['Survived'].values
train.drop(['Survived'],axis=1, inplace=True)

#===============================================================================
# Dropping all but most significant features
#===============================================================================
# train = test[['Pclass','Age','SibSp','Parch']]
# test = test[['PassengerId','Pclass','Age','SibSp','Parch']]
#===============================================================================
# test train split
#===============================================================================
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.values,
                            target,test_size=0.3, random_state=0)


#===============================================================================
# scaling our features
#===============================================================================
# feature scaling is a requirement for regularization, which we will use as our
# first approach to coming up with a better model
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)
test_norm = mms.fit_transform(test.drop(['PassengerId'],axis=1).values)

#===============================================================================
# L1 Regularization & LogisticRegression to find coefficients
#===============================================================================
# NOTE: L1 regularization is typically helpful for high dimensional feature sets
# which is NOT the problem we have

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',C=.1)
lr.fit(X_train_norm, y_train)

print 'Training accuracy:',lr.score(X_train_norm, y_train)
print 'Test accuracy:',lr.score(X_test_norm, y_test)
print 'intercept:', lr.intercept_
print 'coef:', lr.coef_

# building the full model
lr.fit(np.vstack([X_train_norm,X_test_norm]),np.hstack([y_train,y_test]))
lr_pred = lr.predict(test_norm)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": lr_pred
})
submission.to_csv('lr_second_steps.csv', index=False)

#===============================================================================
# Sequential Feature Selection & KNN
#===============================================================================
# import pkgs
# '''this doesnt appear to be working with KNN'''
# # from sklearn.neighbors import KNeighborsClassifier
# # knn = KNeighborsClassifier(n_neighbors=2)
# # sbs = pkgs.SBS(knn, k_features=1)
# # sbs.fit(X_train_norm, y_train)
#
# '''lets try the LR model above'... NOPE SBS still hangs'''
# print 'Running SBS...'
# sbs = pkgs.SBS(LogisticRegression(), k_features=1)
# sbs.fit(X_train_norm,y_train)
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


#===============================================================================
# Random Forest Feature Selection
#===============================================================================
# from sklearn.ensemble import RandomForestClassifier
# from first_forest import process_all_titanic, load_titanic, family_features
#
# train, test = load_titanic()
# X_train, X_test = process_all_titanic(train,test)
# X_train, X_test = family_features(X_train, X_test)
#
# X_train.drop(['PassengerId','Name', 'Ticket','Cabin','Sex','Embarked'],axis=1,inplace=True)
# y = X_train['Survived'].values
# X = X_train.drop(['Survived'],axis=1).values
#
# feat_labels = X_train.drop(['Survived'],axis=1).columns
# forest = RandomForestClassifier(n_estimators=10000,random_state=0, n_jobs=-1)
# forest.fit(X, y)
# importances = forest.feature_importances_
# indicies = np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]-1):
#     print '{0}.\t {1}\t\t {2}'.format(f+1, feat_labels[f],importances[indicies[f]])
#
# import matplotlib.pyplot as plt
# plt.title('Feature Importances')
# plt.bar(range(X_train.shape[1]-1),
#         importances[indicies],
#         color = 'lightblue',
#         align='center')
# plt.xticks(range(X_train.shape[1]-1),
#         feat_labels, rotation=90)
# plt.xlim([-1,X_train.shape[1]-1])
# plt.tight_layout()
# plt.show()
