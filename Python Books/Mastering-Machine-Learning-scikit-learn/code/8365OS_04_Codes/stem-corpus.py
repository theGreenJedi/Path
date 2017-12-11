__author__ = 'gavin'
import csv
from nltk import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
lines = []

# stem training
with open('/home/gavin/mastering-machine-learning/ch4-logistic_regression/movie-reviews/train.tsv', 'rb') as f:
    out = open('/home/gavin/mastering-machine-learning/ch4-logistic_regression/movie-reviews/train-stemmed.tsv', 'wb')
    reader = csv.reader(f, delimiter='\t')
    header = reader.next()
    out.write('PhraseId	SentenceId	Phrase	Sentiment\n')
    for row in reader:
        stemmed = ' '.join([stemmer.stem(token).lower() for token in word_tokenize(row[2])])
        out.write('%s\t%s\t%s\t%s\n' % (row[0], row[1], stemmed, row[3]))
    out.close()

# stem testing
with open('/home/gavin/mastering-machine-learning/ch4-logistic_regression/movie-reviews/test.tsv', 'rb') as f:
    out = open('/home/gavin/mastering-machine-learning/ch4-logistic_regression/movie-reviews/test-stemmed.tsv', 'wb')
    reader = csv.reader(f, delimiter='\t')
    header = reader.next()
    out.write('PhraseId	SentenceId	Phrase\n')
    for row in reader:
        stemmed = ' '.join([stemmer.stem(token).lower() for token in word_tokenize(row[2])])
        out.write('%s\t%s\t%s\n' % (row[0], row[1], stemmed))
    out.close()