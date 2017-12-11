"""
Best score: 0.677
Best parameters set:
	clf__C: 10
	clf__penalty: 'l2'
	vect__max_df: 0.3
	vect__norm: 'l2'
	vect__use_idf: False

SVC:
Best score: 0.677
Best parameters set:
	clf__C: 0.75
	vect__max_df: 0.1

"""
from sklearn.svm.classes import LinearSVC
from ch4.movies import Tokenizer

__author__ = 'gavin'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

tokenizer = Tokenizer()
pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english', max_features=None,
                             ngram_range=(1, 2), use_idf=False, norm='l2', tokenizer=tokenizer)),
    ('clf', LinearSVC())
])
parameters = {
    'vect__max_df': (0.1, 0.5, 0.75),
    # 'vect__max_features': (5000, 10000, None),
    # 'vect__ngram_range': ((1, 1), (1, 2)),
    # 'vect__use_idf': (True, False),
    # 'vect__norm': ('l1', 'l2'),
    # 'clf__loss': ('l1', 'l2'),
    'clf__C': (0.75, 1),
}

if __name__ == "__main__":
    num_jobs = -1
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=num_jobs, verbose=1, scoring='accuracy')
    train_f = '/home/gavin/mastering-machine-learning/ch4-logistic_regression/movie-reviews/train.tsv'
    df = pd.read_csv(train_f,
                 header=0, delimiter='\t')
    X_train = df['Phrase']
    y_train = df['Sentiment'].as_matrix()
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])