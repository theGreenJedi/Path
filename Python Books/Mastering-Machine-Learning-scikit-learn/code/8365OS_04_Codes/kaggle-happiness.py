__author__ = 'gavin'
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split

df = pd.read_csv('/home/gavin/kaggle-happiness/train.csv')

y = df['Happy']
cols = set(df.columns)
cols.remove('Happy')
X = df[list(cols)]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)