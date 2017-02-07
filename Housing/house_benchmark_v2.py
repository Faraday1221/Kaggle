# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# a quick and dirty neural network to benchmark model performance
# no feature engineering or tuning

import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# # import our test and train data
# train = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')
#
# #===================================================================
# # processing
# #===================================================================
# from organize_dummies import compare_categories
#
# # NOTE: we are currently dropping all columns with NULLs
# # we are doing this simply because it is easier FOR NOW
# # than handling the nulls thoughtfully
#
# # standarize the columns across the two datasets
# ref, tst = compare_categories(ref=train.dropna(axis=1),
#                               tst=test.dropna(axis=1),
#                               id='Id',
#                               target='SalePrice')
# # create the arrays for the NN
# X = ref.drop(['Id','SalePrice'],axis=1).values
# Y = ref.SalePrice
# # create the array for predicting Sales Price
# X_test = tst.drop(['Id','SalePrice'],axis=1).values





#===================================================================
# processing on our MODIFIED dataset
#===================================================================
# we are doing most of the processing in another sheet
# this eliminates the need for the preceeding code
train = pd.read_csv('./data/train_mod.csv')
test = pd.read_csv('./data/test_mod.csv')

x = pd.concat([train,test],axis=0)

# NOTE: BAD HANDLING OF DATES UNTIL I GET THEM SORTED!
time_col = ['YearBuilt','YrSold','YearRemodAdd','MoSold','GarageYrBlt']
x[time_col].convert_objects(convert_numeric=True)

# one hot encode my categorical variables
y = pd.get_dummies(x)

X = y[y.Id.isin(train.Id)].drop(['Id','SalePrice'],axis=1).values
Y = y[y.Id.isin(train.Id)]['SalePrice']
X_test = y[y.Id.isin(test.Id)].drop(['Id','SalePrice'],axis=1).values


#===================================================================
# create the MLP neural network
#===================================================================
# create a baseline MLP, 215X input features, 2 hidden layers, regressor
# note 216 is X.shape[1]
def benchmark():
    # create model
    model = Sequential()
    model.add(Dense(300,input_dim=X.shape[1],init='uniform',activation='relu'))
    model.add(Dense(200, input_dim=300, init='uniform', activation='relu'))
    model.add(Dense(100, input_dim=200, init='uniform', activation='relu'))
    model.add(Dense(50, input_dim=100, init='uniform', activation='relu'))
    model.add(Dense(1, init='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducability
seed = 0
np.random.seed(seed)


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

pipe_mlp = Pipeline([

                     ('scl', MinMaxScaler() ),
                    #  ('pca', PCA() ),
                     ('mlp', KerasRegressor(build_fn=benchmark,
                                            nb_epoch=300,
                                            batch_size=50,
                                            verbose=1) )
                    ])

# start = time.time()
# pipe_mlp.fit(X,Y)
# print("total run time: {:.3f}".format(time.time()-start))

#===================================================================
# old code ... here to line 133
#===================================================================

# evaluate model with standardized dataset
# the epochs and batch size are a complete guess
# estimator = KerasRegressor( build_fn=benchmark,
#                             nb_epoch=300,
#                             batch_size=50,
#                             verbose=1)

kfold = KFold(n_splits=5, random_state=seed)

#===================================================================
# train the NN
#===================================================================
# the scale of these results seems way off... why???
# my best guess is they are off by 1e05
start = time.time()
# NOTE: updated estimator to pipe_mlp
results = cross_val_score(pipe_mlp, X, Y, cv=kfold)
# results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: {0:.2f} +/- {1:.2f}".format(results.mean()/1e5,results.std()/1e5))
print("CV run time: {:.3f}".format(time.time()-start))

# the results above were not easy to interpert, instead I'll try a simple check
# start = time.time()
# estimator.fit(X,Y)
# print("total run time: {:.3f}".format(time.time()-start))

#===================================================================
# evaluate the model
#===================================================================
# a poor mans spot check - to ensure I overfit!!
# delta = []
# a,b = X.shape
# for i in range(a):
#     # NOTE: updated with pipeline
#     pred = pipe_mlp.predict(X[i].reshape((1,b)))
#     # pred = estimator.predict(X[i].reshape((1,b)),verbose=0)
#     # print("Estimated: {0}, Actual: {1}, Delta: {2}".format(pred,Y[i],pred-Y[i]))
#     delta.append(np.sqrt((pred-Y[i])**2))
#
# print("RMS Error: {0:.4f} +/- {1:.4f}".format(np.mean(delta),np.std(delta)))

#===================================================================
# generate predictions on the test set
#===================================================================
# submission = pd.DataFrame([tst.Id.astype('str'),
#                           estimator.predict(X_test)],
#                           ['Id','SalePrice']).T

# submission.to_csv('benchmark_newfeatures.csv',index=False)

#===================================================================
# predictions from MOD data
#===================================================================
# NOTE: when using CV as above - we dont fit the model... so predit fails
# submission = pd.DataFrame([test.Id.astype('str'),
#                           pipe_mlp.predict(X_test)],
#                         #   estimator.predict(X_test)],
#                           ['Id','SalePrice']).T
#
# # make sure this is different than the benchmark score
# bench = pd.read_csv('./benchmark.csv')
#
# diff = submission.SalePrice - bench.SalePrice
# print('Difference between mod data & benchmark:')
# print('{0:.1f} +/- {1:.1f}'.format(diff.mean(),diff.std()))
