import sklearn.datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

newgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newgroups.data, newgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


print("Acurácia no teste:", knn.score(X_test, y_test))
