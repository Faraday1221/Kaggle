#!/usr/bin/env python
# coding=utf-8
'''
Here is the code based on the Getting Started with Python Kaggle posts.
https://www.kaggle.com/c/titanic/details/getting-started-with-python

We present two "models". A simple model that predicts based on Gender and a
slightly less simple model that uses a lookup table.

The following can be reused:
- lookup_table: a nice fcn that makes groupbys into lookup tables in 3 lines
- advanced binning with q_bin, retbins and labels. This allows us to bin one dataset and then apply it to another dataset (note floor and ceiling need to be addressed.)
'''
import numpy as np
import pandas as pd
import os

path = os.path.dirname(__file__)+'/data/'
filename = 'train.csv'

# read in the training dataset
train = pd.read_csv(os.path.join(path,filename))

# #looking at our data
# train.info()
# train.describe()

#===============================================================================
# A simple If Then Model: Gender
#===============================================================================
# The first 'model' looks at the liklihood of survival based on gender.

# split our dataset by gender
male = train[train.Sex == 'male']
female = train[train.Sex == 'female']

# calculate the avg chance of survival (0=No, 1=Yes)
male_avg = male.Survived.sum() / float(male.shape[0])
female_avg = female.Survived.sum() / float(female.shape[0])

print '{0:.2f}%  survival rate: {1}'.format(male_avg*100,'male')
print '{0:.2f}%  survival rate: {1}'.format(female_avg*100,'female')

# based on above we can create a highly simple gender model
# first read in the test set
filename = 'test.csv'
test = pd.read_csv(os.path.join(path,filename))

# now we predict if female then survived else not-survived
pred = np.where(test.Sex == 'female',1,0)

#===============================================================================
# A slightly less simple model: Class, Gender and Ticket Price
#===============================================================================
# Next we create a look up table to use as a reference

# reviewing the Fare data, I think we can split based on quantile rather than
# the suggested segmentation, choosing q=4 will roughly match the tutorial
train['Price_Bin'], q_bins = pd.qcut(train.Fare, q=4, precision=0,retbins=True,
labels=False)

# to view our Price Bins
# pd.qcut(train.Fare, q=4, precision=0).value_counts(sort=False)

# create our lookup table, this outputs the mean survival percent for each group
survival_table = train.groupby(['Sex', 'Pclass', 'Price_Bin']).mean()['Survived']

# then we change each parameter to 0 or 1 for a prediction value based on the
# the mean score >= .5 (note: this may need to change shape)
# note this does not show values which do not exist
survival_table = survival_table.apply(lambda x: np.where(x >= .5, 1, 0))
survival_dict = survival_table.to_dict()

# note: there is 1 null in Fare i.e. test.info(), we need to handle this
# we need to acount for NULLS, below is a simple way to do that
test['Fare'].fillna(test.Fare.median(),inplace=True)


#===============================================================================
# Sidebar: filling nulls like a boss
#===============================================================================
# instead of taking a mean we could fill in nulls based on group characteristics
# we can build a lookup dict fairly simply as follows:
fare_table = test.groupby(['Sex','Pclass']).mean()['Fare']
fare_dict = fare_table.to_dict()

# then use this handly helper function to populate values based on our fare_dict
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

# ultimatley we can fill our null values like below using our lookup table
test.Fare.fillna(test.apply(lambda x: lookup_table(x,['Sex','Pclass'],fare_dict)
,axis=1), inplace=True)

# note: there is a line of code just above this that already filled in the nulls

#===============================================================================
# Binning our data
#===============================================================================
# we need to "bin" our train data... note: data is nice, no floor or ceiling
# it is not easy to find this from the documentation but: we can align our two
# categorical datasets by using retbins=True which returns the quantile array
# and using lables = False which outputs the bin number. This trick (along with
# floor and ceiling) will be a helpful way to compare categories across datasets

# HOWEVER: by removing the labels from above our survival table does not account
# for NaN values

test['Price_Bin'] = pd.cut(test.Fare, bins=q_bins, precision=0,labels=False,
include_lowest=True)

#print test.Price_Bin.head()
#print train.Price_Bin.head()

# we wrap this in a function below so make the submission a tad easier
pred = []
for i in range(test.shape[0]):
    x = (test.iloc[i].Sex, test.iloc[i].Pclass, test.iloc[i].Price_Bin)
    pred.append(int(survival_dict[x]))

# pred is our prediction based on our 'slightly less simple model'
# ...and thats all folks!

#===============================================================================
# output format
#===============================================================================
'''the output format should be a csv of PassengerId,Survived.
   see gendermodel.csv for an example'''

def create_submission(df, filename):
    pred = []
    for i in range(df.shape[0]):
        x = (df.iloc[i].Sex, df.iloc[i].Pclass, df.iloc[i].Price_Bin)
        pred.append(int(survival_dict[x]))
    submission = pd.DataFrame({
                    'PassengerId':df['PassengerId'],
                    'Survived':pred})
    submission.to_csv(filename, index=False)

create_submission(test,'first_steps.csv')
