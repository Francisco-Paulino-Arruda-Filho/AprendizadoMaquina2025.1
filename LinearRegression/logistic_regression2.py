from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.datasets
from sklearn.svm import LinearSVC

lr = LogisticRegression()

wine = sklearn.datasets.load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
print("Acurácia no teste:", lr.score(X_test, y_test))   

lr.predict_proba(X_test[:1])

svm = LinearSVC()
X, y = wine.data, wine.target
svm.fit(X, y)
svm.score(X, y)