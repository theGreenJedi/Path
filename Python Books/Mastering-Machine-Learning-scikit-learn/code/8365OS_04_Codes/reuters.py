__author__ = 'gavin'
import glob
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

# Create X and raw y
X = None
labels_for_all_instances = []
vectorizer = HashingVectorizer()
d = {d[0]: d[1:] for d in [l.strip()[9:].split(' ') for l in open('reuters/cats.txt', 'rb') if l.startswith('training')]}
for f in glob.glob('/home/gavin/PycharmProjects/mastering-machine-learning/ch4-logistic_regression/reuters/training/*'):
    text = ' '.join([label.strip() for label in open(f, 'rb')])
    if X is None:
        X = vectorizer.fit_transform([text])
    else:
        X = vstack((X, vectorizer.fit_transform([text])))
    training_id = f[f.rfind('/')+1:]
    labels_for_all_instances.append(d[training_id])

print X.shape

train_len = int(X.shape[0] * .7)
train_len = X.shape[0]-1
X_train = X.tocsc()[:train_len]
X_test = X.tocsc()[train_len:]
for label in set([label for instance in labels_for_all_instances for label in instance]):
    y = [1 if label in instance else 0 for instance in labels_for_all_instances]
    y_train = y[:train_len]
    y_test = y[train_len:]
    print len(y_test)
    print X_test.shape
    classifier = LogisticRegression()
    classifier.fit_transform(X_train, y_train)
    print 'Accuracy for %s: %s' % (label, classifier.score(X_test, y_test))