__author__ = 'gavin'
from nltk import word_tokenize
from nltk.stem import PorterStemmer


class Tokenizer(object):

    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in word_tokenize(doc)]