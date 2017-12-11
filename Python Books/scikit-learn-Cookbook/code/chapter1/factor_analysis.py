# IPython log file


from __future__ import division
import numpy as np
from sklearn import datasets, decomposition
reg = datasets.make_regression(100, 4, 2)
reg
reg
reg.data
import numpy as np
np.column_stack(reg[0], reg[1])
np.column_stack((reg[0], reg[1]))
X = np.column_stack((reg[0], reg[1]))
X.shape
fa
fa = decomposition.FactorAnalysis()
fa.fit(X[-1:])
fa
fa.score
fa.score()
fa.noise_variance_
fa.fit(X)
fa.noise_variance_
X[:-1]
fa.fit(X[:-1])
fa
fa.noise_variance_
X[:-1].shape
X[:,:-1].shape
fa.fit(X[:,:-1])
fa.n_components
fa.noise_variance_
fa.fit(reg[0])
fa.noise_variance_
reg = datasets.make_regression(100, 4, 1)
fa.fit(reg[0])
fa.noise_variance_
fa = decomposition.FactorAnalysis(n_components=2)
fa.fit(reg[0])
fa.noise_variance_
exit()
