

import pandas as pd
import numpy as np
import math
from scipy.stats import skew



#===============================================================================
# Convert Variable Type
#===============================================================================

def convert_quality(df):
    """many of the features are quality based and presented as categorical
       this function converts them to ordinal. inputs the df and returns the
       updated df"""
    # many of the metrics have a generic quality score, we are going to
    # convert this from categorical to ordinal
    quality_scores = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1, 'NA':0}
    quality_list = ['BsmtQual','BsmtCond','ExterQual','ExterCond','HeatingQC',
                'KitchenQual','GarageQual','GarageCond','FireplaceQu','PoolQC']

    for col in quality_list:
        df[col] = df[col].fillna('NA').map(quality_scores)

    # Mapping basement expore into an ordinal feature
    bsmt_expo = {'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0}
    df['BsmtExposure'] = df['BsmtExposure'].fillna('NA').map(bsmt_expo)

    bsmt_type = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0}
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA').map(bsmt_type)
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA').map(bsmt_type)

    # convert Garage Finish into ordinal & zero NULLS
    garage_dict = {'Fin':3,'RFn':2,'Unf':1,'NA':0}
    df['GarageFinish'] = df['GarageFinish'].fillna('NA').map(garage_dict)
    return df

def misc_convert(df):
    """simplifies the eletric categorical, converts MSSubClass from
       numeric to categorical and handles the miscfeature feature.
       inputs the df and returns the updated df"""
    # the idea for this feature is that we would use it as a set of
    # binary indicators (rather than numeric)
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # we can simplify our eletrical categories
    df['Modern_Eletric'] = np.where(df.Electrical=='SBrkr','Modern','Hist')

    # we will turn MisFeature into has shed or not as a binary indicator
    # df.MiscFeature.value_counts()
    df['HasShed'] = np.where(df.MiscFeature == 'Shed',1,0)
    df.drop(['MiscFeature'],axis=1,inplace=True)
    return df

#===============================================================================
# New Features
#===============================================================================

def generate_featues(df):
    """generates new featues based on a manual inspection of the data"""
    # About 1/3 of the houses have 1/2 to 1/3 of their finished living space
    # in the basement. We need a metric for potential bedrooms in the basement
    # we will guess that if there is >1 BA in the basement there is 1 BR
    # >2 BA = 2BR
    df['Bsmt_BR'] = df['BsmtFullBath'].fillna(0)*1.0 + df['BsmtHalfBath'].fillna(0)*0.5
    df['Bsmt_BR'] = df.Bsmt_BR.apply(lambda x: math.floor(x))

    df['Total_BR'] = df['BedroomAbvGr'] + df['Bsmt_BR']

    # total bathrooms
    df['Total_BA'] = df['FullBath'] + df['HalfBath']*0.5 + \
                df['BsmtFullBath'].fillna(0) + df['BsmtHalfBath'].fillna(0)*0.5

    # Rooms other than bedrooms bathrooms and kitchen (as %)
    df['Non_BA_BR'] = (df['BedroomAbvGr'] + df['FullBath'] + \
                df['HalfBath'] + 1) / (df['TotRmsAbvGrd'])

    # ratio of basement finished sf to total sf
    df['BsmtFinSF_Ratio'] = 1 - df['BsmtUnfSF'] / df['TotalBsmtSF']
    df['BsmtFinSF_Ratio'].fillna(0,inplace=True)

    # total living area
    df['TotFinSF'] = df.TotalBsmtSF.fillna(0) - df.BsmtUnfSF.fillna(0) + \
                df.GrLivArea

    # ratio of living space that is above ground
    df['LvgSpaceAbvGr_Ratio'] = 1 - (df['TotalBsmtSF'].fillna(0) - \
                df['BsmtUnfSF'].fillna(0)) / df['TotFinSF']

    # total SF considering porches etc.
    df['All_SF'] = df['TotFinSF'] + df['ScreenPorch'] + df['3SsnPorch'] \
                + df['EnclosedPorch'] + df['OpenPorchSF'] + df['WoodDeckSF']

    # additional SF from porches etc.
    df['PorchSF_Ratio'] =  (df.All_SF - df.TotFinSF) / df.TotFinSF

    # create ordinal values for the Fence... most are NA anyway
    fence_dict = {'GdPrv':2,'GdWo':2,'MnWw':1,'MnPrv':1,'NA':0}
    df['Fence'] = df['Fence'].fillna('NA').map(fence_dict)
    return df


#===============================================================================
# Handling Skew
#===============================================================================

def adjust_skew(df, thresh=0.75):
    """calculates the skew amongst numeric features and log transforms them
       to be more normally distributed when above the threshold value thresh.
       NOTE: this incudes a list of features to explicitly exclude based on
             manual review.

       ALSO: this can affect the target variable and and if so
             will require the target be transformed back after predictions!
             i.e. preds = np.expm1(clf.predict(X_test))
    """
    #log transform skewed numeric features:
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    #compute skewness
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > thresh]
    skewed_feats = skewed_feats.index

    # the approach above needs a bit of tweaking based on our featureset which
    # contains binary indicators as well as categorical mapped to ordinal data
    # in those instances we will simply exclude those columns
    exclude = ['HasShed','PoolQC','ExterQual','ExterCond','Fence']
    skewed_feats = skewed_feats.drop(exclude)

    # log(feature + 1) transform to normalize skew
    df[skewed_feats] = np.log1p(df[skewed_feats])

    return df




#===============================================================================
# Engineering the dates
#===============================================================================
def normalize_dates(df):
    # reference all the years from 1860
    time_col = ['YearBuilt','YrSold','YearRemodAdd','GarageYrBlt']
    df[time_col] = df[time_col] - 1860
    return df

# build date based features
def date_features(df):
    # a few binary columns to designate the recency of a rennovation
    df['new_remod'] = np.where(df.YearRemodAdd == df.YrSold,1,0)
    df['fresh_remod'] = np.where(df.YrSold - df.YearRemodAdd <= 5,1,0)
    # did the house have a remodel yes or no
    df['remod'] = np.where(df.YearRemodAdd == df.YearBuilt, 1,0)

    # age of the house
    df['house_age'] = df.YrSold - df.YearBuilt

    # create quarters
    df['Q1'] = np.where(df.MoSold.isin([1,2,3]),1,0)
    df['Q2'] = np.where(df.MoSold.isin([4,5,6]),1,0)
    df['Q3'] = np.where(df.MoSold.isin([7,8,9]),1,0)
    df['Q4'] = np.where(df.MoSold.isin([10,11,12]),1,0)
    return df


#===============================================================================
# handling missing values
#===============================================================================

def fill_na_one(df):
    """the original logic used to fill na values based on manual review"""
    # things we can just fill with 0
    fill_list = ['BsmtFinSF1','BsmtFinSF2','GarageArea','GarageCars',
                'TotalBsmtSF','Electrical','BsmtUnfSF','BsmtFullBath',
                'BsmtHalfBath','LotFrontage']

    for col in fill_list:
        df[col] = df[col].fillna(0)

    # we are only missing a small number of values here: we fill with the most frequent value
    df['SaleType'].fillna('WD',inplace=True)
    df['Exterior1st'].fillna('VinylSd',inplace=True)
    df['Exterior2nd'].fillna('VinylSd',inplace=True)
    df['Utilities'].fillna('AllPub',inplace=True)
    df['Functional'].fillna('Typ',inplace=True)
    df['MSZoning'].fillna('RL',inplace=True)
    df['MasVnrArea'].fillna(0,inplace=True)
    df['MasVnrType'].fillna('None',inplace=True)
    df['Alley'].fillna('None',inplace=True)

    # we add a new category for GarageType when Null
    df['GarageType'].fillna('None',inplace=True)

    # we add the Year Built to fill in garage year
    df['GarageYrBlt'].fillna(df['YearBuilt'],inplace=True)
    return df



# a helper to select most common object
def select_most_common(df,col):
    "expects a col of objects, returns the most common object in the series"
    return df[col].value_counts().index[0]

def handle_object_nulls(df):
    """parses out object columns from the df, checks if there are any nulls
       then fills nulls with the most common value. returns the whole df with
       the categorical nulls filled"""
    # select only categorical data
    cat_df = df.select_dtypes(exclude=[np.number])

    # check categorical NULLS
    null_check = cat_df.isnull().sum()
    features_w_null = null_check[null_check > 0 ].index

    # if null fill with most common value
    for col in features_w_null:
        df[col] = select_most_common(df,col)

    return df








#===============================================================================
# Extract Test and Train Data
#===============================================================================
def extract_test_train(df,train,test):
    """expects our modified dataframe of the stacked train test data df
       expects the original train and test dataframes,
       returns the split modified dataframes for test and train"""
    # extract our training data
    x = df[df.Id.isin(train.Id)]

    # extrat our test data
    y = df[df.Id.isin(test.Id)]
    # drop SalePrice!
    y = y.drop(['SalePrice'],axis=1)
    return x, y


#===============================================================================
# Testing Pipelines
#===============================================================================
def feature_eng_01(train,test):
    # we combine our test & train data such that we have a fuller picture
    df = pd.concat([train,test],axis=0)

    # original method used to generate modified data set
    df = convert_quality(df)
    df = misc_convert(df)
    df = generate_featues(df)
    df = fill_na_one(df)

    train, test = extract_test_train(df,train,test)
    return train,test


def feature_eng_02(train,test):
    df = pd.concat([train,test],axis=0)
    # same conversions & engineering as eng_01
    df = convert_quality(df)
    df = misc_convert(df)
    df = generate_featues(df)

    # build the date based features
    df = date_features(df)
    df = normalize_dates(df)
    # adjust the skew
    df = adjust_skew(df)
    # handle nulls simply - most freq for object & mean for num
    df = handle_object_nulls(df)
    df = df.fillna(df.mean())
    df = pd.get_dummies(df)

    train, test = extract_test_train(df,train,test)
    return train, test


if __name__ == '__main__':

    path = '/Users/jonbruno/Documents/Python/Kaggle/Housing/data/'
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')

    # our 1st modified feature set
    train01, test01 = feature_eng_01(train,test)

    # our 2nd modified feature set
    train02, test02 = feature_eng_02(train,test)
