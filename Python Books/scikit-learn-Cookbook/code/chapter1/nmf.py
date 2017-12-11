# IPython log file


from __future__ import division
import numpy as np
from sklearn.decomposition import NMF
nmf = NMF()
from sklearn.datasets import load_iris
iris = load_iris()
NMF(iris.data)
nmf.fit_transform(iris.data)
nmf = NMF(n_components=2)
fits = nmf.fit_transform(iris.data)
fits
len(fits)
fits[:5]
nmf.reconstruction_err_
exit()
