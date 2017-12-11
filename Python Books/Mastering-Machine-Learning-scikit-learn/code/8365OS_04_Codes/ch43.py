"""

"""
__author__ = 'gavin'
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, confusion_matrix
import numpy as np
from scipy.sparse import hstack

blacklist = [l.strip() for l in open('insults/blacklist.csv', 'rb')]


def get_counts(documents):
    return np.array([np.sum([c.lower().count(w) for w in blacklist]) for c in documents])


# Note that I cleaned the trianing data by replacing """ with "
train_df = pd.read_csv('insults/train.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(train_df['Comment'], train_df['Insult'])
vectorizer = TfidfVectorizer(max_features=2000, norm='l2', max_df=0.04,
                             ngram_range=(1, 1), stop_words='english', use_idf=False)
X_train = vectorizer.fit_transform(X_train_raw)
#X_train_counts = get_counts(X_train_raw)
#X_train = hstack((X_train, X_train_counts.reshape(len(X_train_counts), 1)))

X_test = vectorizer.transform(X_test_raw)
#X_test_counts = get_counts(X_test_raw)
#X_test = hstack((X_test, X_test_counts.reshape(len(X_test_counts), 1)))
classifier = LogisticRegression(penalty='l1', C=2)
classifier.fit_transform(X_train, y_train)
predictions = classifier.predict(X_test)

print 'accuracy', classifier.score(X_test, y_test)
print 'precision', precision_score(y_test, predictions)
print 'recall', recall_score(y_test, predictions)
print 'auc', roc_auc_score(y_test, predictions)
print confusion_matrix(y_true=y_test, y_pred=predictions)