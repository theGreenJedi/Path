from sklearn.metrics import f1_score
y_true, y_pred = [0, 0, 1, 1], [1, 0, 1, 1]
print f1_score(y_true, y_pred)
y_true, y_pred = [0, 0, 1, 1], [1, 0, 1, 0]
print f1_score(y_true, y_pred)
#>>> 0.8
#>>> 0.5