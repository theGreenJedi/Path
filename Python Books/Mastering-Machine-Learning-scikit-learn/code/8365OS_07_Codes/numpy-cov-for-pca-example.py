"""
[[ 0.61655556  0.61544444]
 [ 0.61544444  0.71655556]]
"""
import numpy as np

# X = [
#     [0.69, 0.49],
#     [-1.31, -1.21],
#     [0.39, 0.99],
#     [0.09, 0.29],
#     [1.29, 1.09],
#     [0.49, 0.79],
#     [0.19, -0.31],
#     [-0.81, -0.81],
#     [-0.31, -0.31],
#     [-0.71, -1.01]
# ]
X = [
    [-0.27, -0.3],
    [1.23, 1.3],
    [0.03, 0.4],
    [-0.67, -0.6],
    [-0.87, -0.6],
    [0.63, 0.1],
    [-0.67, -0.7],
    [-0.87, -0.7],
    [1.33, 1.3],
    [0.13, -0.2]
]
C = np.cov(np.array(X).T)
print C
w, v = np.linalg.eig(C)
print 'w', w
print 'v', v

print 'h', np.dot(np.array(X), v[0].reshape(2, 1))

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X = pca.fit_transform(X)
print 'pca', X