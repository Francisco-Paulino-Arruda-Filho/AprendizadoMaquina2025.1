from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=500, random_state=1111)
rfc.fit(X_train, y_train)
print("Training: {:.2f}".format(rfc.score(accuracy_score(y_train, rfc.predict(X_train)))))

print("Testing: {:.2f}".format(rfc.score(accuracy_score(y_test, rfc.predict(X_test)))))