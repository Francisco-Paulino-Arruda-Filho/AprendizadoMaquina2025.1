from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [1, 2, 3, 4]

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predicts = model.predict(X_test)
print(predicts)