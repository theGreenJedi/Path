import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

X = [
    [.2, .6, .2],
    [.3, .59, .25],
[.4, .58, .4],
[.4, .57, .5],
[.5, .56, .7],
[.6, .55, .6],
[.6, .53, .5],
[.6, .5, .8],
[.6, .49, .75],
[.7, .48, .7],
[.7, .47, .5],
[.8, .45, .65],
[.8, .43, .9],
[.9, 0.41, .85],
[1, 0.4, 1],
]
X = np.array(X)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.draw()
plt.ylim(0, 1.2)
plt.xlim(0, 1.2)
ax.set_zlim(0, 1.2)
plt.show()
