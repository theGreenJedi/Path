from sklearn.feature_extraction.text import CountVectorizer

# The input parameter min_df is a threshold which is used to
# ignore the terms that document frequency less than the
# threshold. By default it is set as 1.
vectorizer = CountVectorizer(min_df=1)
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']

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
# column j in document i.
print(X.toarray())

# Instead of using the count we can also get the binary value
# matrix for the given corpus by setting the binary parameter
# equals True.
vectorizer_binary = CountVectorizer(min_df=1, binary=True)
X_binary = vectorizer_binary.fit_transform(corpus)

# The value of a[i, j] == 1 means that the word corresponding to
# column j is present in document i
print(X_binary.toarray())
