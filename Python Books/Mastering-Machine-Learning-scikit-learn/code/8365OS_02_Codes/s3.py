from sklearn.preprocessing import PolynomialFeatures
X_train = [[6], [8], [10], [14], [18]]
X_test = [[6],  [8],   [11], [16]]
featurizer = PolynomialFeatures(degree=2)
X_train = featurizer.fit_transform(X_train)
X_test = featurizer.transform(X_test)
print X_train
print X_test