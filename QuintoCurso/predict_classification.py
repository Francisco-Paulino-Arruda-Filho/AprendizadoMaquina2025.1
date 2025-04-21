from sklearn.ensemble import RandomForestClassifier

X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(random_state=1111, n_estimators=50, max_depth=10)
rfc.fit(X_train, y_train)
rfc.predict(X_test)
rfc.predict_proba(X_test)
rfc.get_params()
rfc.score(X_test, y_test)