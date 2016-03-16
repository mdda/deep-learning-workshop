import sklearn.linear_model

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_val, y_val)