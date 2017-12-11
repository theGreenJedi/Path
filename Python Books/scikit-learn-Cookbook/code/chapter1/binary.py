# IPython log file


from __future__ import division
import numpy as np
from sklearn import preprocessing
from skleanr import datasets
from sklearn import datasets
boston = datasets.load_boston()
print boston.DESCR
boston.target[:5]
boston.target.mean()
from sklearn import preprocessing
get_ipython().magic(u'pinfo preprocessing.binarize')
new_target = preprocessing.binarize(boston.target, threshold=boston.mean())
new_target = preprocessing.binarize(boston.target, threshold=boston.target.mean())
new_target[:5]
(boston.target[:5] > boston.target.mean()).astype(int)
bin = preprocessing.Binarizer(boston.target.mean())
boston_prime = bin.fit_transform(boston.target)
boston_prime[:5]
from scipy.sparse import csc
get_ipython().magic(u'pinfo csc')
from scipy.sparse import coo
get_ipython().magic(u'pinfo coo')
get_ipython().magic(u'pinfo coo.coo_matrix')
coo.coo_matrix(np.random.binomial(1, .25, 100))
spar = coo.coo_matrix(np.random.binomial(1, .25, 100))
preprocessing.binarize(spar, threshold=-1)
exit()
