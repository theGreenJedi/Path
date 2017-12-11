from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']

# The input parameter min_df is a threshold which is used to
# ignore the terms that document frequency less than the
# threshold. By default it is set as 1.
vectorizer = TfidfVectorizer(min_df=1)

# fit_transform method basically Learn the vocabulary dictionary
# and return term-document matrix.
X = vectorizer.fit_transform(corpus)

# Each term found by the analyzer during the fit is assigned a
# unique integer index corresponding to a column in the resulting
# matrix.
print(vectorizer.get_feature_names())

# The numerical features can be extracted by the method toarray
# It returns a matrix in the form of (n_corpus, n_features)
# The columns correspond to vectorizer.get_feature_names(). The
# value of a[i, j] is basically the count of word correspond to
# column j in document i
print(X.toarray())
