"""
Best score: 0.992
Best parameters set:
	clf__C: 7.0
	clf__penalty: 'l2'
	vect__max_df: 0.5
	vect__max_features: None
	vect__ngram_range: (1, 2)
	vect__norm: 'l2'
	vect__use_idf: True
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import precision_score, recall_score, confusion_matrix

__author__ = 'gavin'
import pandas as pd

df = pd.read_csv('sms/sms.csv')

X_train_r, X_test_r, y_train, y_test = train_test_split(df['message'], df['label'])

vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, ngram_range=(1, 1), norm='l2', use_idf=True)
X_train = vectorizer.fit_transform(X_train_r)
X_test = vectorizer.transform(X_test_r)
classifier = LogisticRegression(penalty='l2', C=7)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print 'score', classifier.score(X_test, y_test)
print 'precision', precision_score(y_test, predictions)
print 'recall', recall_score(y_test, predictions)
print confusion_matrix(y_test, predictions)
