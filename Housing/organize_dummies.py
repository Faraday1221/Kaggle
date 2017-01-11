# pandas.get_dummies is great except when your train and test data
# dont have the same categorical values, then if you try pandas.get_dummies on
# both your test and train will have different columns
# this code is intended to standarize the columns by ensuring test and train
# have the same columns

# compare_categories is the function that addresses the issue above

import numpy as np
import pandas as pd

def dummies(df):
    "return a dummies df and the set of column names"
    x = pd.get_dummies(df)
    return x, set(x.columns)

def add_zeros(df,name):
    "add a new column to the df with all zeros"
    df[name] = np.zeros((df.shape[0],1))
    return df

# https://goo.gl/7aWn42
def order_id_target(df,col,position):
    "reorder a single col > str in the df by position > int"
    cols = df.columns.tolist()
    cols.insert(position, cols.pop(cols.index(col)))
    return df.ix[:, cols]

def compare_categories(ref, tst, id=None, target=None):
    """description:
       addresses the issue where we want to use dummies to represent
       categorical values as binary indicators across with test and train
       data sets. The function returns both test and train datasets with
       binary indicators for all categorical variables, zero padded when
       that variable is not in the data set, such that both test and train
       will have the same number of columns in the same order.

       When id and target are used, the id column is moved to the first
       position and the target column is moved to the last position

       params:
        - ref: expects a dataframe of the train dataset
        - tst: expects a dataframe of the test dataset
        - id:  expects a string with the Id column name
        - target:   expects a string with the target column name
        """
    r,r_set = dummies(ref)
    t,t_set = dummies(tst)

    # add categories to test, when in ref and not in test
    for name in r_set.difference(t_set):
        t = add_zeros(t,name)
    # add categories to ref, when in test and not in ref
    for name in t_set.difference(r_set):
        r = add_zeros(r,name)

    # sort the columns lexographically
    r.sort_index(axis=1,inplace=True)
    t.sort_index(axis=1,inplace=True)

    # reposition the id and target columns
    if (id != None) and (target != None):
        r = order_id_target(r,id,0)
        r = order_id_target(r,target,len(r.columns))
        t = order_id_target(t,id,0)

    return r,t
