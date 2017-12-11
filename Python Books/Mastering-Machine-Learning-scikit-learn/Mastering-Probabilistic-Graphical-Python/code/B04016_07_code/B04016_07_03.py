from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

# The dataset used in this example is the 20 newsgroups dataset.
# The 20 Newsgroups data set is a collection of
# approximately 20,000 newsgroup documents, partitioned (nearly)
# evenly across 20 different newsgroups. It will be
# automatically downloaded, then cached.
# For our simple example we are only going to use 4 news group
categories = ['alt.atheism',
              'talk.religion.misc',
              'comp.graphics',
              'sci.space']

# Loading training data
data_train = fetch_20newsgroups(subset='train',
                                categories=categories,
                                shuffle=True,
                                random_state=42)

# Loading test data
data_test = fetch_20newsgroups(subset='test',
                               categories=categories,
                               shuffle=True,
                               random_state=42)
y_train, y_test = data_train.target, data_test.target

# It can be changed to "count" if we want to use count vectorizer
feature_extractor_type = "hashed"
if feature_extractor_type == "hashed":
    # To convert the text documents into numerical features,
    # we need to use a feature extractor. In this example we
    # are using HashingVectorizer as it would be memory
    # efficient in case of large datasets
    vectorizer = HashingVectorizer(stop_words='english')

    # In case of HashingVectorizer we don't need to fit
    # the data, just transform would work.
    X_train = vectorizer.transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
elif feature_extractor_type == "count":
    # The other vectorizer we can use is CountVectorizer with
    # binary=True. But for CountVectorizer we need to fit
    # transform over both training and test data as it
    # requires the complete vocabulary to create the matrix
    vectorizer = CountVectorizer(stop_words='english', binary=True)

# First fit the data
vectorizer.fit(data_train.data + data_test.data)

# Then transform it
X_train = vectorizer.transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# alpha is additive (Laplace/Lidstone) smoothing parameter (0 for
# no smoothing).
clf = BernoulliNB(alpha=.01)

# Training the classifier
clf.fit(X_train, y_train)

# Predicting results
y_predicted = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_predicted)
print("accuracy: %0.3f" % score)

