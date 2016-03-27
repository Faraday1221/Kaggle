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
