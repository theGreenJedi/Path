################# Sample 1 #################
"""
>>> import pandas as pd
>>> df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
>>> print df.head()
      0                                                  1
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
[5 rows x 2 columns]
>>> print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
>>> print 'Number of ham messages:', df[df[0] == 'ham'][0].count()
Number of spam messages: 747
Number of ham messages: 4825
"""
import pandas as pd
df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
print df.head()
print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
print 'Number of ham messages:', df[df[0] == 'ham'][0].count()


################# Sample 2 #################
"""
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split
>>> from sklearn.metrics.metrics import precision_score, recall_score, confusion_matrix

>>> df = pd.read_csv('sms/sms.csv')
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> predictions = classifier.predict(X_test)

>>> for i, prediction in enumerate(predictions[:5]):
>>>     print X_test_raw[i], 'prediction:', prediction
Were trying to find a Chinese food place around here prediction: 0
all the lastest from Stereophonics, Marley, Dizzee Racal, Libertines and The Strokes! Win Nookii games with Flirt!! Click TheMob WAP Bookmark or text WAP to 82468 prediction: 1
We have pizza if u want prediction: 0
I like you peoples very much:) but am very shy pa. prediction: 0
Dun need to use dial up juz open da browser n surf... prediction: 0
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split

df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

for i, prediction in enumerate(predictions[:5]):
    print X_test_raw[i], 'prediction:', prediction


################# Sample 3 #################
"""
>>> from sklearn.metrics import confusion_matrix
>>> import matplotlib.pyplot as plt

>>> y_test = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
>>> y_pred = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
>>> confusion_matrix = confusion_matrix(y_test, y_pred)
>>> print(confusion_matrix)
>>> plt.matshow(confusion_matrix)
>>> plt.title('Confusion matrix')
>>> plt.colorbar()
>>> plt.ylabel('True label')
>>> plt.xlabel('Predicted label')
>>> plt.show()
[[4 1]
 [2 3]]
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_test = []
y_pred = []
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


################# Sample 4 #################
"""
>>> from sklearn.metrics import accuracy_score
>>> y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
>>> print 'Accuracy:', accuracy_score(y_true, y_pred)
Accuracy: 0.5
"""
from sklearn.metrics import accuracy_score
y_pred, y_true = [0, 1, 1, 0], [1, 1, 1, 1]
print accuracy_score(y_true, y_pred)


################# Sample: Evaluating the SMS Classifier #################
"""
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_score

>>> df = pd.read_csv('data/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> scores = cross_val_score(classifier, X_train, y_train, cv=5)
>>> print np.mean(scores), scores
Accuracy 0.956217208018 [ 0.96057348  0.95334928  0.96411483  0.95454545  0.94850299]
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print np.mean(scores), scores


################# Sample 6 #################
"""
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.cross_validation import train_test_split, cross_val_score

>>> df = pd.read_csv('data/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression()
>>> classifier.fit(X_train, y_train)
>>> precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
>>> print 'Precision', np.mean(precisions), precisions
>>> recall = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
>>> print 'Recall', np.mean(recall), recall
Precision 0.992137651822 [ 0.98717949  0.98666667  1.          0.98684211  1.        ]
Recall 0.677114261885 [ 0.7         0.67272727  0.6         0.68807339  0.72477064]
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print 'Precision', np.mean(precisions), precisions
recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print 'Recall', np.mean(recalls), recalls


################# Sample: F1 Score #################
"""
>>> f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
>>> print 'F1', np.mean(f1s), f1s
F1 0.80261302628 [ 0.82539683  0.8         0.77348066  0.83157895  0.7826087 ]
"""
f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print 'F1', np.mean(f1s), f1s


################# Sample: ROC AUC #################
"""

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('data/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


################# Sample 8 #################
"""
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model.logistic import LogisticRegression
>>> from sklearn.metrics import roc_curve, auc
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.cross_validation import train_test_split

>>> df = pd.read_csv('sms/sms.csv')
>>> X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
>>> vectorizer = TfidfVectorizer()
>>> X_train = vectorizer.fit_transform(X_train_raw)
>>> X_test = vectorizer.transform(X_test_raw)
>>> classifier = LogisticRegression().fit_transform(X_train, y_train)
>>> clf = LogisticRegression()
>>> clf.fit_transform(X_train, y_train)
>>> predictions = clf.predict_proba(X_test)
>>> false_positive_rate, recall, threhsolds = roc_curve(y_test, predictions[:, 1])
>>> roc_auc = auc(false_positive_rate, recall)
>>> plt.title('Receiver Operating Characteristic')
>>> plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
>>> plt.legend(loc='lower right')
>>> plt.plot([0, 1], [0, 1], 'r--')
>>> plt.xlim([0.0, 1.0])
>>> plt.ylim([0.0, 1.0])
>>> plt.ylabel('Recall')
>>> plt.xlabel('Fall-out')
>>> plt.show()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

df = pd.read_csv('sms/sms.csv')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression().fit_transform(X_train, y_train)
clf = LogisticRegression()
clf.fit_transform(X_train, y_train)
predictions = clf.predict_proba(X_test)
false_positive_rate, recall, threhsolds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


################# Sample 9 #################
"""
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    df = pd.read_csv('data/sms.csv')
    X, y, = df['message'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Precision:', precision_score(y_test, predictions)
    print 'Recall:', recall_score(y_test, predictions)
Fitting 3 folds for each of 1536 candidates, totalling 4608 fits
[Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done  50 jobs       | elapsed:    4.0s
[Parallel(n_jobs=-1)]: Done 200 jobs       | elapsed:   16.9s
[Parallel(n_jobs=-1)]: Done 450 jobs       | elapsed:   36.7s
[Parallel(n_jobs=-1)]: Done 800 jobs       | elapsed:  1.1min
[Parallel(n_jobs=-1)]: Done 1250 jobs       | elapsed:  1.7min
[Parallel(n_jobs=-1)]: Done 1800 jobs       | elapsed:  2.5min
[Parallel(n_jobs=-1)]: Done 2450 jobs       | elapsed:  3.4min
[Parallel(n_jobs=-1)]: Done 3200 jobs       | elapsed:  4.4min
[Parallel(n_jobs=-1)]: Done 4050 jobs       | elapsed:  7.7min
[Parallel(n_jobs=-1)]: Done 4608 out of 4608 | elapsed:  8.5min finished
Best score: 0.983
Best parameters set:
	clf__C: 10
	clf__penalty: 'l2'
	vect__max_df: 0.5
	vect__max_features: None
	vect__ngram_range: (1, 2)
	vect__norm: 'l2'
	vect__stop_words: None
	vect__use_idf: True
Accuracy: 0.989956958393
Precision: 0.988095238095
Recall: 0.932584269663
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    df = pd.read_csv('data/sms.csv')
    X, y, = df['message'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Precision:', precision_score(y_test, predictions)
    print 'Recall:', recall_score(y_test, predictions)
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)
    df = pd.read_csv('data/sms.csv')
    X, y, = df['message'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print 'Accuracy:', accuracy_score(y_test, predictions)
    print 'Precision:', precision_score(y_test, predictions)
    print 'Recall:', recall_score(y_test, predictions)


################# Sample 9 #################
"""
>>> import pandas as pd
>>> df = pd.read_csv('movie-reviews/train.tsv', header=0, delimiter='\t')
>>> print df.count()
PhraseId      156060
SentenceId    156060
Phrase        156060
Sentiment     156060
dtype: int64
>>> print df.head()
   PhraseId  SentenceId                                             Phrase  \
0         1           1  A series of escapades demonstrating the adage ...
1         2           1  A series of escapades demonstrating the adage ...
2         3           1                                           A series
3         4           1                                                  A
4         5           1                                             series

   Sentiment
0          1
1          2
2          2
3          2
4          2

[5 rows x 4 columns]
>>> print df['Phrase'].head(10)
0    A series of escapades demonstrating the adage ...
1    A series of escapades demonstrating the adage ...
2                                             A series
3                                                    A
4                                               series
5    of escapades demonstrating the adage that what...
6                                                   of
7    escapades demonstrating the adage that what is...
8                                            escapades
9    demonstrating the adage that what is good for ...
Name: Phrase, dtype: object
>>> print df['Sentiment'].describe()
count    156060.000000
mean          2.063578
std           0.893832
min           0.000000
25%           2.000000
50%           2.000000
75%           3.000000
max           4.000000
Name: Sentiment, dtype: float64
>>> print df['Sentiment'].value_counts()
2    79582
3    32927
1    27273
4     9206
0     7072
dtype: int64
>>> print df['Sentiment'].value_counts()/df['Sentiment'].count()
2    0.509945
3    0.210989
1    0.174760
4    0.058990
0    0.045316
dtype: float64
"""
import pandas as pd
df = pd.read_csv('data/train.tsv', header=0, delimiter='\t')
print df.count()
print df.head(10)
print df['Phrase'].head()
print df['Sentiment'].describe()
print df['Sentiment'].value_counts()
print df['Sentiment'].value_counts()/df['Sentiment'].count()


################# Sample: Multi-Class Classification of Movie Review Sentiments #################
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def main():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10),
    }
    df = pd.read_csv('data/train.tsv', header=0, delimiter='\t')
    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

if __name__ == '__main__':
    main()

Fitting 3 folds for each of 24 candidates, totalling 72 fits
[Parallel(n_jobs=3)]: Done   1 jobs       | elapsed:    3.3s
[Parallel(n_jobs=3)]: Done  50 jobs       | elapsed:  1.1min
[Parallel(n_jobs=3)]: Done  68 out of  72 | elapsed:  1.9min remaining:    6.8s
[Parallel(n_jobs=3)]: Done  72 out of  72 | elapsed:  2.1min finished
Best score: 0.620
Best parameters set:
	clf__C: 10
	vect__max_df: 0.25
	vect__ngram_range: (1, 2)
	vect__use_idf: False
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def main():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__use_idf': (True, False),
        'clf__C': (0.1, 1, 10),
    }
    df = pd.read_csv('data/train.tsv', header=0, delimiter='\t')
    X, y = df['Phrase'], df['Sentiment'].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

if __name__ == '__main__':
    main()


################# Sample 11 #################
# Multi-Class Classification Performance Metrics
"""
>>> predictions = grid_search.predict(X_test)
>>> print 'Accuracy:', accuracy_score(y_test, predictions)
>>> print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
>>> print 'Classification Report:', classification_report(y_test, predictions)
Accuracy: 0.611521209791
Confusion Matrix: [[  443  1576  1367    93     4]
 [  248  3892  9021   534    16]
 [   54  1699 35836  2026    67]
 [    5   409  9024  6693   440]
 [    1    88  1112  2529   853]]
Classification Report:              precision    recall  f1-score   support

          0       0.59      0.13      0.21      3483
          1       0.51      0.28      0.36     13711
          2       0.64      0.90      0.75     39682
          3       0.56      0.40      0.47     16571
          4       0.62      0.19      0.29      4583

avg / total       0.59      0.61      0.57     78030
"""
predictions = grid_search.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, predictions)
print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
print 'Classification Report:', classification_report(y_test, predictions)


################# Sample 12 #################
# Applying Multi-label Classification
"""

"""


################# Sample 13 #################
# Multi-Label Classification Performance Metrics
"""
>>> import numpy as np
>>> from sklearn.metrics import hamming_loss
>>> print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
0.0
>>> print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
0.25
>>> print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))
0.5
>>> print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
1.0
>>> print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
0.75
>>> print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))
0.5
"""
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_similarity_score
print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
print hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [1.0, 1.0]]))
print jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]))