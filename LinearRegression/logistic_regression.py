from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.datasets

lr = LogisticRegression()

newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()
X, y = newsgroups.data, newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Acurácia no teste:", lr.score(X_test, y_test))
