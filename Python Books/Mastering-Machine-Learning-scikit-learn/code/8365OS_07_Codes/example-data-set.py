__author__ = 'gavin'
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np

## for figure 2

def princomp(A):
    M = (A-mean(A.T,axis=1)).T
    [latent, coeff] = linalg.eig(cov(M))
    score = dot(coeff.T, M)
    return coeff, score, latent

X = [
    [.2, .2],
    [.4, .4],
    [.6, .6],
    [.7, .7],
    [1, 1],
    [.4, .5],
    [.6, .5],
    [.6, .8],
    [.5, .7],
    [.8, .65],
    [.3, .25],
    [.8, .9],
    [.9, .85],
    [.6, .75],
    [.7, .5],
]

from numpy import mean, cov, cumsum, dot, linalg, array, rank

A = np.array(X).T
print A
# A = array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
#             [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])
coeff, score, latent = princomp(A.T)

m = mean(A, axis=1)
# plt.plot([-1, -coeff[0, 0]*2]+m[0], [-1, -coeff[0, 1]*2]+m[1], '--k')
plt.plot([-1, -coeff[0, 0]*2]+m[0], [-1, -coeff[0, 1]*2]+m[1], '--k')
plt.plot([0.6, 0], [0, 1.0], ':r')
a = [
    [0, -coeff[0, 0]*2]+m[0],
    [0, -coeff[0, 1]*2]+m[1]
    ]


plt.plot(A[0, :], A[1, :], 'ob')
plt.xlim(0, 1.2)
plt.ylim(0, 1.2)
plt.show()

# X = np.array(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# m = np.mean(X, axis=1)
# print 'mean', m
#
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# X_red = pca.fit_transform(X)
# print pca.components_
#
# X = np.array(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.plot([0, pca.components_[0][0]+m[0]], [0, pca.components_[1][0]+m[1]])
# plt.show()


## for gigure 3

__author__ = 'gavin'
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np

X = [
    [.2, .2],
    [.4, .4],
    [.6, .6],
    [.7, .7],
    [1, 1],
    [.4, .5],
    [.6, .5],
    [.6, .8],
    [.5, .7],
    [.8, .65],
    [.3, .25],
    [.8, .9],
    [.9, .85],
    [.6, .75],
    [.7, .5],
]

data = np.array(X)
xData = data[:, 0]
yData = data[:, 1]

mu = data.mean(axis=0)
data = data - mu
eigenvectors, eigenvalues, V = np.linalg.svd(
    data.T, full_matrices=False)
projected_data = np.dot(data, eigenvectors)
sigma = projected_data.std(axis=0).mean()
print(eigenvectors)

def annotate(ax, name, start, end, facecolor):
    arrow = ax.annotate(name,
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(facecolor=facecolor, width=2.0))
    return arrow

fig, ax = plt.subplots()
ax.scatter(xData, yData)
ax.set_aspect('equal')
counter = 0
facecolors = ['red', 'green']
for axis in eigenvectors:
    print 'mu', mu
    print sigma
    annotate(ax, '', mu, mu + sigma * axis, facecolors[counter])
    counter += 1
plt.show()
