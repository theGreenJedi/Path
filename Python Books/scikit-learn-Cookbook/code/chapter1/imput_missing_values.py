# IPython log file


from __future__ import division
import numpy as np
from sklearn import datasets, preprocessing
iris = datasets.load_iris()
iris_X = iris.data
missing_mask = np.random.binomial(1, .1, iris_X.shape).astype(bool)
missing_mask[:10]
iris_X[missing_mask] = np.nan
iris_X[:10]
missing_mask[:10]
missing_mask[:5]
iris_X[:5]
impute = preprocessing.Imputer()
impute = preprocessing.Imputer(missing_values=np.nan)
impute.fit(iris_X)
iris_X_prime = impute.transform(iris_X)
iris_X_prime
iris_X_prime
iris_X_prime[:5]
iris_X_prime[5]
iris_X_prime[6]
iris_X_prime[4]
iris_X_prime[3]
iris_X_prime[3, 1]
iris_X_prime[3, 0]
iris_X[3, 0]
get_ipython().magic(u'ls ')
iris_X
impute = preprocessing.Imputer(missing_values=np.nan, strategy='median')
iris_X_prime = impute.transform(iris_X)
impute = preprocessing.Imputer(missing_values=np.nan, strategy="median")
iris_X_prime = impute.transform(iris_X)
iris_X
iris_X_prime = impute.transform(iris_X)
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]
get_ipython().magic(u'pinfo preprocessing.Imputer')
iris_X
np.isnan(iris_X)
iris_X[np.isnan(iris_X)] = '-'
iris_X[np.isnan(iris_X)] = -1
iris_X[:5]
impute = preprocessing.Imputer(missing_values=-1)
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]
get_ipython().magic(u'pinfo preprocessing.Imputer')
iris.feature_names
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)
import pandas as pd
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)
iris_df.head()
get_ipython().magic(u'pinfo iris_df.where')
iris_X[missing_mask] = np.nan
iris_df
iris_df.fillna(iris.df.mean())
iris_df.fillna(iris_df.mean())
iris_df.fillna(iris_df.mean()).head(5)
df.mean()
iris_df.mean()
iris_df.max()
iris_df.fillna(iris_df.max()).head(5)
iris_df.fillna(iris_df.max())['sepal length (cm)'].head(5)
iris_df.fillna(iris_df.mean())['sepal length (cm)'].head(5)
exit()
