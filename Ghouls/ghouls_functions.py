
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class ColumnExtractor(TransformerMixin,BaseEstimator):
    """takes in a dataframe, parses it by columns and returns an np array"""
    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self

class GetDummies(TransformerMixin,BaseEstimator):
    """I hate LabelEncoder and OneHotEncoder this is my workaround"""
    def __init__(self):
        pass
#         self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
#         return pd.get_dummies(X[self.columns]).values #this assumed we were passing in a df, X is a np array
        return pd.get_dummies(X).values

    def fit(self, X, y=None, **fit_params):
        return self
# I want to use the sklearn polynomial features but dont want to deal with labels!
# http://stackoverflow.com/questions/36728287/sklearn-preprocessing-polynomialfeatures-how-to-keep-column-names-headers-of
def PolynomialFeatures_labeled(input_df,power):
    '''Basically this is a cover for the sklearn preprocessing function.
    The problem with that function is if you give it a labeled dataframe, it ouputs an unlabeled dataframe with potentially
    a whole bunch of unlabeled columns.

    Inputs:
    input_df = Your labeled pandas dataframe (list of x's not raised to any power)
    power = what order polynomial you want variables up to. (use the same power as you want entered into pp.PolynomialFeatures(power) directly)

    Ouput:
    Output: This function relies on the powers_ matrix which is one of the preprocessing function's outputs to create logical labels and
    outputs a labeled pandas dataframe
    '''
    from sklearn import preprocessing as pp

    poly = pp.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df


def LinearSVC_2D_plot(clf,feature_1,feature_2):
    """plots a 2D LinearSVC model based on 2 features
       e.g. clf = LinearSVC()
            feature_1 = 'has_soul'
            feature_2 = 'hair_length'"""

    clf.fit(X[[feature_1,feature_2]].values, Y.values)
    print('score: {:.3f}'.format(clf.score(X[[feature_1,feature_2]].values, Y.values)))

    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[feature_1].values.min() - 1, X[feature_1].values.max() + 1
    y_min, y_max = X[feature_2].values.min() - 1, X[feature_2].values.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[feature_1].values, X[feature_2].values, c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
#     plt.xlim(-.25,1.25)
#     plt.ylim(-.25, 1.25)
    plt.xticks(())
    plt.show()
