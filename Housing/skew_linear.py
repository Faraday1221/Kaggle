"""this is a nice start to evaluating linear models with sklearn...
   it would be great to expand on this in a more formal way to evaluate
   a larger suite of linear models
"""
# there is a beautiful linear modeling approach found here:
# https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/
# regularized-linear-models/discussion

# all of the following is from that post... with minor adjustments


import pandas as pd
import numpy as np

from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, LassoCV
from sklearn.linear_model import RidgeCV, ElasticNet, LassoLarsCV
from sklearn.model_selection import cross_val_score

import matplotlib
import matplotlib.pyplot as plt

# we will use the modified features dataset
train = pd.read_csv("./data/train_mod.csv")
test = pd.read_csv("./data/test_mod.csv")

# concat our data together to make preprocessing easier
all_data = pd.concat((train.drop(['Id','SalePrice'],axis=1),
                      test.drop(['Id'],axis=1)))

#===============================================================================
# plot of distributions
#===============================================================================
def plot_dists():
    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
    prices.hist()

#===============================================================================
# Handling skew
#===============================================================================
# for normally distributed data the skew should be about 0
# the threshold for "redistributing" our data is set as 0.75 by the author
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#compute skewness
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

# the approach above needs a bit of tweaking based on our featureset which
# contains binary indicators as well as categorical mapped to ordinal data
# in those instances we will simply exclude those columns
exclude = ['HasShed','PoolQC','MSSubClass','ExterQual','ExterCond','Fence']
skewed_feats = skewed_feats.drop(exclude)

# log(feature + 1) transform to normalize skew
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#===============================================================================
# ...a hair more preprocessing
#===============================================================================
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


#===============================================================================
# Building & Testing Linear Models - RIDGE
#===============================================================================

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y,
                  scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


def ridgeCV():
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
    return cv_ridge


def plot_ridge_alpha():
    # make a series of models and check their rmse
    cv_ridge = ridgeCV()
    # plot the rmse
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    cv_ridge.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    # return the min rmse to get an idea of model performance
    return cv_ridge.min()

print('Ridge rmse_cv min: {:3f}'.format( np.min(ridgeCV())))
# the main tuning parameter for Ridge is alpha
# when alpha is too large the model is over regularized and has increasing bias
# when alpha is too small the model is has increasing variance from overfitting

#===============================================================================
# Building & Testing LASSO
#===============================================================================

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
# print('Lasso rmse_cv mean: {:3f}'.format(rmse_cv(model_lasso).mean()))
print('Lasso rmse_cv min: {:3f}'.format(np.min(rmse_cv(model_lasso))))

# Nice! The lasso performs even better so we'll just use this one to predict on the test set. Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# we can also look at the most important features
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

def plot_lasso_features():
    matplotlib.rcParams['figure.figsize'] = (4.0, 6.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")


#let's look at the residuals as well:
def plot_lasso_residuals():
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

    preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")

#===============================================================================
# Preditions & Submission
#===============================================================================
# since Lasso outperformed Ridge we will use that model

# we have to unpack the lasso model since we transformed the sales price
preds = np.expm1(model_lasso.predict(X_test))

# then make our submission dataframe
submission = pd.DataFrame({'Id':test.Id,'SalePrice':preds})
