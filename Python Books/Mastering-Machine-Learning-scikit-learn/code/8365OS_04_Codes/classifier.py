"""

"""
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# TODO refactor this to not use pandas
# TODO load one file and use train plot split
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test_with_solutions.csv")

y_train = np.array(train_data.Insult)
X_train = np.array(train_data.Comment)

#cv = CountVectorizer()
#cv.fit(X_train)

#X_train = cv.transform(X_train).tocsr()
X_test = np.array(test_data.Comment)
y_test = np.array(test_data.Insult)
#X_test = cv.transform(comments_test)

#pipeline = Pipeline([
#    ('vect', CountVectorizer(max_df=0.5, ngram_range=(1, 1))),
#    ('tfidf', TfidfTransformer()),
#    ('clf', SGDClassifier(alpha=0.00001, penalty='l2'))
#])
#pipeline.fit(X_train, y_train)
cv = CountVectorizer(max_df=0.5, ngram_range=(1, 1))
cv.fit(X_train)
X_train = cv.transform(X_train).tocsr()
X_test = cv.transform(X_test)
tt = TfidfTransformer()
#tt.fit(X_train)
X_train = tt.fit_transform(X_train)
#tt.fit(X_test)
X_test = tt.fit_transform(X_test)
#print 'logistic regression', pipeline.score(X_test, y_test)

clf = SGDClassifier()
clf.fit(X_train, y_train)
print 'logreg', clf.score(X_test, y_test)

svm = LinearSVC()
svm.fit(X_train, y_train)
print 'svm score', svm.score(X_test, y_test)