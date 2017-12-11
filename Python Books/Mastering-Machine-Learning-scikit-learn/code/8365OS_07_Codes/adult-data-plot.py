import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('data/adult.data', header=None)
y = df[14]
X = df[range(0, 14)]


def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return data, vecData, vec


X, _, _ = one_hot_dataframe(X, range(0, 14), replace=True)
X.fillna(-1, inplace=True)

labels = []
for i in y:
    if i == ' <=50K':
        labels.append(0)
    else:
        labels.append(1)

pca = PCA(n_components=2)
print 'fitting pca'
X = pca.fit_transform(X)

print 'plotting'
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()