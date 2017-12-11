# IPython log file


from __future__ import division
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
from sklearn import decomposition
get_ipython().magic(u'pinfo decomposition')
iris_X
pca = decomposition.PCA()
pca.score
pca
iris_pca = pca.fit_transform(iris_X)
iris_pca[:5]
import numpy as np
np.set_printoptions(precision=3)
iris_pca[:5]
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()
iris_X_prime = pca.fit_transform(iris_X)
iris_X_prime
pca = decomposition.PCA(n_components=2)
iris_X_prime = pca.fit_transform(iris_X)
iris_X_prime
iris_X_prime.shape
from matplotlib import pyplot as plt
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_X_prime[:,0], iris_X_prime[:, 1], c=iris.target)
f.savefig('pca.png')
get_ipython().system(u'open pca.png')
ax.set_title("PCA 2 Components")
f.savefig('pca.png')
get_ipython().system(u'open pca.png')
pca.explained_variance_ratio_.sum()
pca = decomposition.PCA(n_components=.98)
iris_X_prime = pca.fit_transform(iris_X)
pca.explained_variance_ratio_.sum()
iris_X_prime.shape
exit()
