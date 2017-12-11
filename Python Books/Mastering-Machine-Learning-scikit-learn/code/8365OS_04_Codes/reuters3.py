import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.metrics import hamming_loss, precision_score, recall_score, accuracy_score
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.metrics import confusion_matrix

__author__ = 'gavin'
import pandas as pd
import json

good_categories = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'wheat', 'ship', 'corn',
                   'money-supply', 'dlr', 'sugar', 'oilseed', 'coffee', 'gnp', 'gold', 'veg-oil', 'soybean', 'nat-gas']

articles = json.loads(open('reuters/reuters-21578-json/reuters.json', 'rb').read())

good_articles = []
for article_id in articles:
    if 'topics' not in articles[article_id] or 'body' not in articles[article_id]:
        continue
    article_topics = articles[article_id]['topics']
    intersection = len([val for val in article_topics if val in good_categories])
    if intersection > 0:
        good_articles.append(articles[article_id])



print len(good_articles)
print good_articles[0]
print '\n'

y = []
X = []
for article in good_articles:
    y.append(article['topics'])
    X.append(article['body'])

X_train, X_test, y_train_all, y_test_all = train_test_split(X, y)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report

y_true_all = []
predictions_all = []
for label in good_categories[:3]:
    print 'label', label
    y_train = [1 if label in instance else 0 for instance in y_train_all]
    y_test = [1 if label in instance else 0 for instance in y_test_all]
    y_true_all.append(y_test)
    classifier = LogisticRegression()
    classifier.fit_transform(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_all.append(predictions)
    print classification_report(y_test, predictions)
    print confusion_matrix(y_test, predictions)
    print 'precision', precision_score(y_test, predictions)
    print 'recall', recall_score(y_test, predictions)
    print 'accuracy', accuracy_score(y_test, predictions)
    print '\n'


y_true_all = np.array(y_true_all)
predictions_all = np.array(predictions_all)

print hamming_loss(y_true_all, predictions_all)
