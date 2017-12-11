"""
Best score: 0.992
Best parameters set:
	clf__C: 7.0
	clf__penalty: 'l2'
	vect__max_df: 0.5
	vect__max_features: None
	vect__ngram_range: (1, 2)
	vect__norm: 'l2'
	vect__use_idf: True
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


pipeline = Pipeline([
    ('vect', TfidfVectorizer(max_df=0.05, stop_words='english')),
    ('clf', LogisticRegression())
])
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (10000, 13000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (3.0, 5.0, 7.0),
}

if __name__ == "__main__":
    num_jobs = -1
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=num_jobs, verbose=1, scoring='roc_auc')
    df = pd.read_csv('sms/sms.csv')
    grid_search.fit(df['message'], df['label'])
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

