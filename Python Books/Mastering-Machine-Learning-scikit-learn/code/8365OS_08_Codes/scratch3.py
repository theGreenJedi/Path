from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.4, 0.6],
    [0.5, 0.2],
    [0.7, 0.9]
])
X_test = np.array([
    [0.7, 0.8]
])

Y = np.array([1, 1, 1, 0])
h = 0.02


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
fig = plt.figure()

for e in range(1, 7):
    print '\nStarting epoch', e
    clf = Perceptron(n_iter=e, verbose=5).fit(X, Y)
    print clf.intercept_, clf.coef_
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # fig.add_subplot(1, 5, e)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.title('Epoch %s' % e)

    if clf.score(X, Y) == 1:
        print 'converged in epoch', e
        break
    plt.show()