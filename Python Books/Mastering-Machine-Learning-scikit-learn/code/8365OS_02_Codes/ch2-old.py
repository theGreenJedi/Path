


################# Sample 2 #################
"""
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> X = [[6], [8], [10], [14],   [18]]
>>> y = [[7], [9], [13], [17.5], [18]]
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> print 'Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2)
>>> print 'R-squared: %.4f' % model.score(X, y)
Residual sum of squares: 1.75
R-squared: 0.9100
"""
import numpy as np
from sklearn.linear_model import LinearRegression
X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8],  [9],   [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]
model = LinearRegression()
model.fit(X, y)
print 'R-squared: %.4f' % model.score(X_test, y_test)



################# Sample 3 #################
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
model = LinearRegression()
model.fit(X, y)
print 'Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2)
print 'Variance score: %.2f' % model.score(X, y)

################# Sample 4 #################
"""
>>> from numpy.linalg import inv
>>> from numpy import dot, transpose
>>> X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> print dot(inv(dot(transpose(X), X)), dot(transpose(X), y))
[[ 1.1875    ]
 [ 1.01041667]
 [ 0.39583333]]
"""
from numpy.linalg import inv
from numpy import dot, transpose
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
print dot(inv(dot(transpose(X), X)), dot(transpose(X), y))

################# Sample 5 #################
"""
>>> from numpy.linalg import lstsq
>>> X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> print lstsq(X, y)[0]
[[ 1.1875    ]
 [ 1.01041667]
 [ 0.39583333]]
"""
from numpy.linalg import lstsq
X = [[1, 6, 2], [1, 8, 1], [1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
print lstsq(X, y)[0]


################# Sample 6 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
>>> y = [[7],    [9],    [13],    [17.5],  [18]]
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
>>> y_test = [[11],   [8.5],  [15],    [18],    [11]]
>>> predictions = model.predict(X_test)
>>> for i, prediction in enumerate(predictions):
>>>     print 'Predicted: %s, Target: %s' % (prediction, y_test[i])
>>> print 'R-squared: %.2f' % model.score(X_test, y_test)
Predicted: [ 10.0625], Target: [11]
Predicted: [ 10.28125], Target: [8.5]
Predicted: [ 13.09375], Target: [15]
Predicted: [ 18.14583333], Target: [18]
Predicted: [ 13.3125], Target: [11]
R-squared: 0.77
"""
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7],    [9],    [13],    [17.5],  [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11],   [8.5],  [15],    [18],    [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s, Target: %s' % (prediction, y_test[i])
print 'R-squared: %.2f' % model.score(X_test, y_test)

################# Sample 21.1 #################
"""
>>> import pandas as pd
>>> df = pd.read_csv('winequality-red.csv', sep=';')
>>> df.describe()

                pH    sulphates      alcohol      quality
count  1599.000000  1599.000000  1599.000000  1599.000000
mean      3.311113     0.658149    10.422983     5.636023
std       0.154386     0.169507     1.065668     0.807569
min       2.740000     0.330000     8.400000     3.000000
25%       3.210000     0.550000     9.500000     5.000000
50%       3.310000     0.620000    10.200000     6.000000
75%       3.400000     0.730000    11.100000     6.000000
max       4.010000     2.000000    14.900000     8.000000
"""
import pandas as pd
df = pd.read_csv('winequality-red.csv', sep=';')
df.describe()

################# Sample 21.2 #################
"""
>>> import matplotlib.pylab as plt
>>> plt.scatter(df['alcohol'], df['quality'])
>>> plt.xlabel('Alcohol')
>>> plt.ylabel('Quality')
>>> plt.title('Alcohol Against Quality')
>>> plt.show()
"""
import matplotlib.pylab as plt
plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

################# Sample 23 #################
"""
>>> from sklearn.linear_model import LinearRegression
>>> import pandas as pd
>>> import matplotlib.pylab as plt
>>> from sklearn.cross_validation import train_test_split

>>> df = pd.read_csv('wine/winequality-red.csv', sep=';')
>>> X = df[list(df.columns)[:-1]]
>>> y = df['quality']
>>> X_train, X_test, y_train, y_test = train_test_split(X, y)

>>> regressor = LinearRegression()
>>> regressor.fit(X_train, y_train)
>>> y_predictions = regressor.predict(X_test)
>>> print 'R-squared:', regressor.score(X_test, y_test)
0.345622479617
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split

df = pd.read_csv('wine/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print 'R-squared:', regressor.score(X_test, y_test)


################# Sample 23 #################
"""
>>> import pandas as pd
>>> from sklearn import cross_validation
>>> from sklearn.linear_model import LinearRegression
>>> df = pd.read_csv('data/winequality-red.csv', sep=';')
>>> X = df[list(df.columns)[:-1]]
>>> y = df['quality']
>>> regressor = LinearRegression()
>>> scores = cross_validation.cross_val_score(regressor, X, y, cv=5)
>>> print scores.mean(), scores
0.290041628842 [ 0.13200871  0.31858135  0.34955348  0.369145    0.2809196 ]
"""
import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
df = pd.read_csv('data/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
regressor = LinearRegression()
scores = cross_validation.cross_val_score(regressor, X, y, cv=5)
print scores.mean(), scores


################# Sample 25 #################
"""
>>> def evaluate(y_pred, y_true, threshold=1):
>>>     errors = y_pred - y_true
>>>     correct = 0
>>>     for error in errors:
>>>         if abs(error) < threshold:
>>>             correct += 1
>>>     return correct / len(y_true)

>>> for threshold in [1, 0.5, 0.25]:
>>>     print 'Accuracy:', evaluate(y_test, y_predictions, threshold), 'for threshold:', threshold

Accuracy: 0.8475 for threshold: 1
Accuracy: 0.58 for threshold: 0.5
Accuracy: 0.315 for threshold: 0.25
"""
def evaluate(y_pred, y_true, threshold=1):
    errors = y_pred - y_true
    correct = 0
    for error in errors:
        if abs(error) < threshold:
            correct += 1
    return correct / len(y_true)

for threshold in [1, 0.5, 0.25]:
    print 'Accuracy:', evaluate(y_test, y_predictions, threshold), 'for threshold:', threshold

################# Sample 25 #################
"""
>>> X = df[['alcohol', 'volatile acidity', 'sulphates', 'chlorides']]
>>> X_train = X[:1000]
>>> X_test = X[1000:]
>>> y_train = y[:1000]
>>> y_test = y[1000:]
>>> regressor = LinearRegression()
>>> regressor.fit(X_train, y_train)
>>> print 'R-squared:', regressor.score(X_test, y_test)
>>> y_predictions = regressor.predict(X_test)
>>> for threshold in [1, 0.5, 0.25]:
>>>     print 'Accuracy:', evaluate(y_test, y_predictions, threshold), 'for threshold:', threshold
R-squared: 0.353505295256
Accuracy: 0.893155258765 for threshold: 1
Accuracy: 0.587646076795 for threshold: 0.5
Accuracy: 0.30550918197 for threshold: 0.25
"""
X = df[['alcohol', 'volatile acidity', 'sulphates', 'chlorides']]
X_train = X[:1000]
X_test = X[1000:]
y_train = y[:1000]
y_test = y[1000:]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print 'R-squared:', regressor.score(X_test, y_test)
y_predictions = regressor.predict(X_test)
for threshold in [1, 0.5, 0.25]:
    print 'Accuracy:', evaluate(y_test, y_predictions, threshold), 'for threshold:', threshold