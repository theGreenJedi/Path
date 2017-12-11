from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.metrics import classification_report

__author__ = 'gavin'
import glob
from sklearn.linear_model import LogisticRegression

articles = []
labels = []
d = {d[0]: d[1:] for d in [l.strip()[9:].split(' ') for l in open('reuters/cats.txt', 'rb') if l.startswith('training')]}
for f in glob.glob('/home/gavin/PycharmProjects/mastering-machine-learning/ch4-logistic_regression/reuters/training/*'):
    training_id = f[f.rfind('/')+1:]
    articles.append(' '.join([label.strip() for label in open(f, 'rb')]))
    labels.append(d[training_id])

vectorizer = TfidfVectorizer()
train_len = int(len(articles) * .7)
X_train = vectorizer.fit_transform(articles[:train_len])
X_test = vectorizer.transform(articles[train_len:])

for label in set([label for instance in labels for label in instance][:3]):
    y = [1 if label in instance else 0 for instance in labels]
    print y
    y_train = y[:train_len]
    y_test = y[train_len:]
    classifier = LogisticRegression()
    classifier.fit_transform(X_train, y_train)
    predictions = classifier.predict(X_test)
    print y_test
    print predictions
    print classification_report(y_test, predictions)

