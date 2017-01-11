# ultimately I want to cluster these houses so lets build an unspervised
# model to test out how we cluster prior to implementing in the regression
# model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===================================================================
# Feature Selection - Numeric Values only
#===================================================================

# before we cluster, lets build some intuition around what our most important
# features look like:
# e.g. http://scikit-learn.org/stable/modules/feature_selection.html

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest

# read in our training data
X = pd.read_csv('./data/train.csv')

# select only the numeric columns
x = X.select_dtypes(include=[np.number])
# test the best predictors of accurate regression 'SalePrice'
kbest = SelectKBest(f_regression,k=15)
bst = kbest.fit_transform(x.drop('SalePrice',axis=1).fillna('0'),x.SalePrice)

# to see the socres and pvalues from our selector use
# kbest.scores_  # F_scores
# kbest.pvalues_ # P values

# while the above is helpful we really need to unpack the results
# to extract the columns and build our insights build our insights
# this function will help (note the np array returns rows in order best @ 0idx)

def KBestColumns(raw_df,best_array):
    """raw_df expects a df used to feature selection
       best_array expects a np array, the output of feature selection
          with k best features
       KBestColumns returns a list of the best columns and prints their
          importance
    """
    best_col = []
    for i in range(best_array.shape[1]):
        col_nm = raw_df.T[raw_df.T == best_array[:,i]].dropna().index.values[0]
        best_col.append(col_nm)
        print("{0}:\t {1}".format(i+1,col_nm))
    return best_col


#===================================================================
# Clustering
#===================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# normalize our data
minmax = MinMaxScaler()

# lets see how well we do with the scaled vs. raw data
# the raw data had distortion of ~ 1000 @ 8 and ~900 @ 15
# x_scaled = minmax.fit_transform(x.fillna('0'))

# even with the top 25 features (all low pvalues) we have a major improvement
# our cluster 1 is distortion of 700, effectively we have eliminated a lot of
# noise, the general trend seems similar.
x_scaled = minmax.fit_transform(bst)

# we will then cluster our data and watch as it -hopefully- converges
# one of the challanges with KMeans is that you have to specify a number of
# clusters apriori, we will use the elbow method to see what represents an
# approproate number of clusters @k=8 dist ~260 @k15 dist ~210

# python machine learning p.320
distortions = []
for i in range(1,41):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(x_scaled)
    distortions.append(km.inertia_)

plt.plot(range(1,41), distortions, marker='o')
plt.title('Elbow Plot PML p.320')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
