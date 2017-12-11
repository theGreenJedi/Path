"""
From the Perceptron to Support Vector Machines
"""
################# Figure 09_01 #################
"""

"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(121)
X = [0.7, 1.0, 2.8, 4.1]
y = [4.5, 3.0, 2.4, 0.35]
X_tr = [0.8, 1.1, 3.3, 1.8]
y_tr = [4.2, 3.1, 2.8, 1.9]
X_tr2 = [0.6, 0.9, 3.0, 4.0]
y_tr2 = [3.4, 2.8, 2.3, 0.4]
X2 = []
y2 = []
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.scatter(X_tr, y_tr)
plt.scatter(X_tr2, y_tr2, marker='x')
fit = np.polyfit(X, y, 3)
fit_fn = np.poly1d(fit)
Xmesh = np.arange(0, 5, 0.1)
plt.xticks([])
plt.yticks([])
plt.plot(Xmesh, fit_fn(Xmesh))
plt.title('X')
plt.subplot(122)
X = [0.7, 1.0, 2.8, 4.1]
y = [4.5, 3.0, 2.4, 0.35]
X_tr = [0.8, 1.1, 4.2, 1.8]
y_tr = [4.2, 3.1, 1, 2.9]
X_tr2 = [0.6, 0.9, 2.1, 3.4]
y_tr2 = [2.4, 2.2, 1.3, 0.2]
X2 = []
y2 = []
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.scatter(X_tr, y_tr)
plt.scatter(X_tr2, y_tr2, marker='x')
plt.plot([0, 4], [4, 0])
plt.xticks([])
plt.yticks([])
plt.title(r'$\phi(X)$')
plt.show()

################# Figure 09_02 #################
"""

"""
import matplotlib
matplotlib.use('Qt4Agg')
import numpy as np
import matplotlib.pyplot as plt

x1 = [2.9, 3.3, 3.5, 3.7]
y1 = [3.9, 3.6, 3.1, 3.2]

x2 = [0.8, 1.0, 1.3, 1.5]
y2 = [2.0, 2.4, 2.3, 2.0]

plt.scatter(x1, y1, marker='o')
plt.scatter(x2, y2, marker='x')
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.plot([0, 5], [3, 3], '--')
plt.plot([0, 4], [7, 0], ':')
plt.plot([0, 8], [5, 0])
plt.xticks([])
plt.yticks([])
plt.title(r'Decision Boundaries')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()


################# Figure: Rings to Linearly Separable #################
"""

"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = 1
cx = []
cy = []
range = np.arange(0, 10, 0.4)
for i in range:
    cx.append(r * np.cos(i))
    cy.append(r * np.sin(i))


plt.scatter(cx, cy, marker='.', s=200)
plt.xticks([])
plt.yticks([])
bx = np.random.rand(10) - 0.5
by = np.random.rand(10) - 0.5
plt.scatter(bx, by, marker='x', s=200)
plt.show()

t_cx = []
t_cy = []
t_cz = []
for i, x in enumerate(cx):
    t_cx.append(x**2)
    t_cy.append(cy[i]**2)
    t_cz.append(math.sqrt(2) * x * cy[i])

t_bx = []
t_by = []
t_bz = []
for i, x in enumerate(bx):
    t_bx.append(x**2)
    t_by.append(by[i]**2)
    t_bz.append(math.sqrt(2) * x * by[i])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(t_bx, t_by, t_bz, marker='x')
ax.scatter(t_cx, t_cy, t_cz, marker='.')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()


################# Figure 09_03 #################
"""

"""
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

x1 = [1.9, 3.3, 3.5, 4.2]
y1 = [3.9, 3.6, 3.1, 4.2]
x2 = [0.8, 1.0, 1.3, 2.5]
y2 = [2.0, 2.4, 2.3, 2.0]
plt.scatter(x1, y1, marker='o')
plt.scatter(x2, y2, marker='x')
plt.annotate('A', xy=(4.2, 4.2), xytext = (-5, 5), textcoords='offset points', ha='right', va='bottom')
plt.annotate('B', xy=(3.5, 3.1), xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
plt.annotate('C', xy=(1.9, 3.9), xytext = (-5, 5), textcoords='offset points', ha='right', va='bottom')
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.plot([0, 4], [7, 0])
plt.xticks([])
plt.yticks([])
plt.show()


################# Figure 09_04 #################
"""
>>> import matplotlib.pyplot as plt
>>> from sklearn.datasets import fetch_mldata
>>> import matplotlib.cm as cm

>>> digits = fetch_mldata('MNIST original', data_home='data/mnist').data

>>> counter = 1
>>> for i in range(1, 4):
>>>     for j in range(1, 6):
>>>         plt.subplot(3, 5, counter)
>>>         plt.imshow(digits[(i - 1) * 8000 + j].reshape((28, 28)), cmap=cm.Greys_r)
>>>         plt.axis('off')
>>>         counter += 1
>>> plt.show()
"""
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import matplotlib.cm as cm

digits = fetch_mldata('MNIST original', data_home='data/mnist').data

counter = 1
for i in range(1, 4):
    for j in range(1, 6):
        plt.subplot(3, 5, counter)
        plt.imshow(digits[(i - 1) * 8000 + j].reshape((28, 28)), cmap=cm.Greys_r)
        plt.axis('off')
        counter += 1
plt.show()

################# Sample 09_01 #################
"""
>>> from sklearn.datasets import fetch_mldata
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.preprocessing import scale
>>> from sklearn.cross_validation import train_test_split
>>> from sklearn.svm import SVC
>>> from sklearn.grid_search import GridSearchCV
>>> from sklearn.metrics import classification_report


>>> if __name__ == '__main__':
>>>     data = fetch_mldata('MNIST original', data_home='data/mnist')
>>>     X, y = data.data, data.target
>>>     X = X/255.0*2 - 1
>>>     # X = scale(X)
>>>     X_train, X_test, y_train, y_test = train_test_split(X, y)
>>>     # clf = SVC(kernel='rbf', C=2.8, gamma=.0073)
>>>     pipeline = Pipeline([
>>>         ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
>>>     ])
>>>     print X_train.shape
>>>     parameters = {
>>>         'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
>>>         'clf__C': (0.1, 0.3, 1, 3, 10, 30),
>>>     }
>>>     grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1, scoring='accuracy')
>>>     grid_search.fit(X_train[:10000], y_train[:10000])
>>>     print 'Best score: %0.3f' % grid_search.best_score_
>>>     print 'Best parameters set:'
>>>     best_parameters = grid_search.best_estimator_.get_params()
>>>     for param_name in sorted(parameters.keys()):
>>>         print '\t%s: %r' % (param_name, best_parameters[param_name])
>>>     predictions = grid_search.predict(X_test)
>>>     print classification_report(y_test, predictions)
Fitting 3 folds for each of 30 candidates, totalling 90 fits
[Parallel(n_jobs=2)]: Done   1 jobs       | elapsed:  7.7min
[Parallel(n_jobs=2)]: Done  50 jobs       | elapsed: 201.2min
[Parallel(n_jobs=2)]: Done  88 out of  90 | elapsed: 304.8min remaining:  6.9min
[Parallel(n_jobs=2)]: Done  90 out of  90 | elapsed: 309.2min finished
Best score: 0.966
Best parameters set:
	clf__C: 3
	clf__gamma: 0.01
             precision    recall  f1-score   support

        0.0       0.98      0.99      0.99      1758
        1.0       0.98      0.99      0.98      1968
        2.0       0.95      0.97      0.96      1727
        3.0       0.97      0.95      0.96      1803
        4.0       0.97      0.98      0.97      1714
        5.0       0.96      0.96      0.96      1535
        6.0       0.98      0.98      0.98      1758
        7.0       0.97      0.96      0.97      1840
        8.0       0.95      0.96      0.96      1668
        9.0       0.96      0.95      0.96      1729

avg / total       0.97      0.97      0.97     17500
"""
from sklearn.datasets import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


if __name__ == '__main__':
    data = fetch_mldata('MNIST original', data_home='data/mnist')
    X, y = data.data, data.target
    # X = X/255.0*2 - 1
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # clf = SVC(kernel='rbf', C=2.8, gamma=.0073)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
    # TODO is refit true by default?
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)


################# Sample 09_02 #################
"""
import os
import numpy as np
import mahotas as mh
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    X = []
    y = []
    for path, subdirs, files in os.walk('data/English/Img/GoodImg/Bmp/'):
        for filename in files:
            f = os.path.join(path, filename)
            target = filename[3:filename.index('-')]
            img = mh.imread(f, as_grey=True)
            if img.shape[0] <= 30 or img.shape[1] <= 30:
                continue
            img_resized = mh.imresize(img, (30, 30))
            if img_resized.shape != (30, 30):
                img_resized = mh.imresize(img_resized, (30, 30))
            X.append(img_resized.reshape((900, 1)))
            y.append(target)
    X = np.array(X)
    X = X.reshape(X.shape[:2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)

Fitting 3 folds for each of 30 candidates, totalling 90 fits
[Parallel(n_jobs=3)]: Done   1 jobs       | elapsed:  1.6min
[Parallel(n_jobs=3)]: Done  50 jobs       | elapsed: 34.8min
[Parallel(n_jobs=3)]: Done  86 out of  90 | elapsed: 69.4min remaining:  3.2min
[Parallel(n_jobs=3)]: Done  90 out of  90 | elapsed: 71.6min finished
Best score: 0.559
Best parameters set:
	clf__C: 3
	clf__gamma: 0.03
             precision    recall  f1-score   support

        001       0.00      0.00      0.00         6
        002       1.00      0.20      0.33         5
        ...
        061       0.00      0.00      0.00         4
        062       0.00      0.00      0.00         4

avg / total       0.56      0.58      0.53       532
"""
import os
import numpy as np
import mahotas as mh
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    X = []
    y = []
    for path, subdirs, files in os.walk('data/English/Img/GoodImg/Bmp/'):
        for filename in files:
            f = os.path.join(path, filename)
            target = filename[3:filename.index('-')]
            img = mh.imread(f, as_grey=True)
            if img.shape[0] <= 30 or img.shape[1] <= 30:
                continue
            img_resized = mh.imresize(img, (30, 30))
            if img_resized.shape != (30, 30):
                img_resized = mh.imresize(img_resized, (30, 30))
            X.append(img_resized.reshape((900, 1)))
            y.append(target)
    X = np.array(X)
    X = X.reshape(X.shape[:2])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)
