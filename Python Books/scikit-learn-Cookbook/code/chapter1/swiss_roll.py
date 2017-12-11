import sklearn.datasets as d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np

swiss_roll = d.make_swiss_roll()

f = plt.figure(figsize=(8, 4))

ax = f.add_subplot(111, projection='3d')
ax.set_title("A swiss roll with Y flattened.")

colors = np.array(['r', 'g', 'b'])
X = swiss_roll[0]

ax.scatter(X[:, 0], np.zeros_like(X[:, 1]), X[:, 2], alpha=0.75)

f.savefig("swiss_roll.png")
