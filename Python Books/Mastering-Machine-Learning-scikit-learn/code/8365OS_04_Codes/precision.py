from sklearn.metrics import precision_score
y_true = [0, 0, 1, 1]
y_pred = [1, 0, 1, 1]
print precision_score(y_true, y_pred)
y_true = [0, 0, 1, 1]
y_pred = [1, 0, 1, 0]
print precision_score(y_true, y_pred)
0.666666666667
0.5