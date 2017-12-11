"""

"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier()

# TODO refactor this to not use pandas
# TODO load one file and use train plot split
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test_with_solutions.csv")

y_train = np.array(train_data.Insult)
X_train = np.array(train_data.Comment)

X_test = np.array(test_data.Comment)
y_test = np.array(test_data.Insult)

cv = CountVectorizer()
cv.fit(X_train)
X_train = cv.transform(X_train)
X_test = cv.transform(X_test)

classifier.fit(X_train, y_train)
print 'raw counts', classifier.score(X_test, y_test)

tt = TfidfTransformer()
X_train = tt.fit_transform(X_train)
X_test = tt.fit_transform(X_test)
classifier.fit(X_train, y_train)
print 'tfidf transformed', classifier.score(X_test, y_test)