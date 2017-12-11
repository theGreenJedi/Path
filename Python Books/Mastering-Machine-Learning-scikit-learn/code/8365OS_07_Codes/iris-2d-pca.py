import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
print 'fitting pca'
X = pca.fit_transform(X)

markers = []
for i in y:
    if i == 0:
        markers.append('x')
    elif i == 0:
        markers.append('.')
    else:
        markers.append('D')

print 'plotting'
plt.scatter(X[:, 0], X[:, 1], c=y, marker=markers)
plt.show()