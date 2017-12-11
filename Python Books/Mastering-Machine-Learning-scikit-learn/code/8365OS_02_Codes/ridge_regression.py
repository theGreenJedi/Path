from sklearn.linear_model LinearRegression, Ridge, Lasso
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_te
