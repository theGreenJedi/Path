import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0.2, 0.1],
    [0.4, 0.6],
    [0.5, 0.2],
    [0.7, 0.9]
])

y = [0, 0, 0, 1]

markers = ['.', 'x']
plt.scatter(X[:3, 0], X[:3, 1], marker='.', s=400)
plt.scatter(X[3, 0], X[3, 1], marker='x', s=400)
plt.xlabel('Proportion of the day spent sleeping')
plt.ylabel('Proportion of the day spent being grumpy')
plt.title('Kittens and Adult Cats')
# plt.plot([0, -2.72], [-3.3, 0])
# plt.plot([0, -13.33], [-3.6363, 0])
plt.plot([0, -30], [4.286, 0])
plt.show()

"""
0.6, 0.02, -0.14

0.02a - 0.14b = -0.6

0, 4.286
-30, 0

"""