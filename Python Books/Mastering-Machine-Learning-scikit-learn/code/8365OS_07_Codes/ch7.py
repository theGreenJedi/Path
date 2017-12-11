"""

"""
__author__ = 'gavinhackeling@gmail.com'

################# Sample 1 #################
"""
>>> import numpy as np
>>> print np.linalg.eig(np.array([[3, 2], [1, 2]]))[0]
[ 4.  1.]
"""
import numpy as np
print np.linalg.eig(np.array([[3, 2], [1, 2]]))[0]

W, V = np.linalg.eig(np.array([[3, 2], [1, 2]]))


################# Sample 2 #################
"""
>>> import numpy as np
>>> print np.linalg.eig(np.array([[3, 2], [1, 2]]))[1]
[[ 0.89442719 -0.70710678]
 [ 0.4472136   0.70710678]]
"""
import numpy as np
print np.linalg.eig(np.array([[3, 2], [1, 2]]))[1]


################# Figure 1#################
"""

"""
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np

A = np.array([[2, 1], [1, 2]])
plt.scatter(A[:, 0], A[:, 1], marker='.', s=200)
plt.plot(A[0], A[1])
plt.show()


################# Sample: Face Recognition #################
"""
>>> from os import walk, path
>>> import numpy as np
>>> import mahotas as mh
>>> from sklearn.cross_validation import train_test_split
>>> from sklearn.cross_validation import cross_val_score
>>> from sklearn.preprocessing import scale
>>> from sklearn.decomposition import PCA
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import classification_report

>>> X = []
>>> y = []
>>> for dir_path, dir_names, file_names in walk('data/att-faces/orl_faces'):
>>>     for fn in file_names:
>>>         if fn[-3:] == 'pgm':
>>>             image_filename = path.join(dir_path, fn)
>>>             X.append(scale(mh.imread(image_filename, as_grey=True).reshape(10304).astype('float32')))
>>>             y.append(dir_path)

>>> X = np.array(X)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y)
>>> pca = PCA(n_components=150)
>>> X_train_reduced = pca.fit_transform(X_train)
>>> X_test_reduced = pca.transform(X_test)
>>> print 'The original dimensions of the training data were', X_train.shape
>>> print 'The reduced dimensions of the training data are', X_train_reduced.shape
>>> classifier = LogisticRegression()
>>> accuracies = cross_val_score(classifier, X_train_reduced, y_train)
>>> print 'Cross validation accuracy:', np.mean(accuracies), accuracies
>>> classifier.fit(X_train_reduced, y_train)
>>> predictions = classifier.predict(X_test_reduced)
>>> print classification_report(y_test, predictions)

The original dimensions of the training data were (300, 10304)
The reduced dimensions of the training data are (300, 150)
Cross validation accuracy: 0.833841819347 [ 0.82882883  0.83        0.84269663]
             precision    recall  f1-score   support

data/att-faces/orl_faces/s1       1.00      1.00      1.00         2
data/att-faces/orl_faces/s10       1.00      1.00      1.00         2
data/att-faces/orl_faces/s11       1.00      0.60      0.75         5
...
data/att-faces/orl_faces/s9       1.00      1.00      1.00         2

avg / total       0.92      0.89      0.89       100
"""
from os import walk, path
import numpy as np
import mahotas as mh
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = []
y = []
for dir_path, dir_names, file_names in walk('data/att-faces/orl_faces'):
    for fn in file_names:
        if fn[-3:] == 'pgm':
            image_filename = path.join(dir_path, fn)
            X.append(scale(mh.imread(image_filename, as_grey=True).reshape(10304).astype('float32')))
            y.append(dir_path)

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=150)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print 'The original dimensions of the training data were', X_train.shape
print 'The reduced dimensions of the training data are', X_train_reduced.shape
classifier = LogisticRegression()
accuracies = cross_val_score(classifier, X_train_reduced, y_train)
print 'Cross validation accuracy:', np.mean(accuracies), accuracies
classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print classification_report(y_test, predictions)