__author__ = 'gavin'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.feature_extraction import DictVectorizer
from sklearn.manifold import Isomap, LocallyLinearEmbedding


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


df = pd.read_csv('data/bank.csv', header=0, delimiter=';')

labels = set(df.columns.values)
labels.remove('y')
X_raw = df[list(labels)]
X_train, _, _ = one_hot_dataframe(X_raw, ['job', 'marital', 'education', 'default', 'housing',
                                 'loan', 'contact', 'month', 'poutcome'], replace=True)
y_train = [1 if i == 'yes' else 0 for i in df.y]

reductions = []
pca = PCA(n_components=2)
reductions.append(pca.fit_transform(X_train, y_train))
lda = LDA(n_components=2)
reductions.append(lda.fit_transform(X_train, y_train))
isomap = Isomap(n_components=2)
reductions.append(isomap.fit_transform(X_train, y_train))
lle = LocallyLinearEmbedding(n_components=2, method='standard')
reductions.append(lle.fit_transform(X_train, y_train))

for reduced_X in reductions:
    plt.figure()
    red_x = []
    red_y = []
    blue_x = []
    blue_y = []
    green_x = []
    green_y = []

    for i in range(len(reduced_X)):
        if y_train[i] == 0:
            red_x.append(reduced_X[i][0])
            red_y.append(reduced_X[i][1])
        elif y_train[i] == 1:
            blue_x.append(reduced_X[i][0])
            blue_y.append(reduced_X[i][1])

    plt.scatter(red_x, red_y, c='r')
    plt.scatter(blue_x, blue_y, c='b')
    plt.show()
