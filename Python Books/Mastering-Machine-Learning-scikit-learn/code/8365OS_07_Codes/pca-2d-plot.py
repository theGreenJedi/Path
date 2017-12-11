import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd

df = pd.read_csv('sms/sms.csv')
y = df.label
X = df.message

from sklearn.feature_extraction.text import TfidfVectorizer

print 'vectorizing'
vectorizer = TfidfVectorizer(max_df=0.5, min_df=3)
X = vectorizer.fit_transform(X)
print X.shape
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

pca = PCA(n_components=2)
print 'fitting pca'
X = pca.fit_transform(X.toarray())

print 'plotting'
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()