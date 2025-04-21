from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sum(abs(test_predictions - y_test)) / len(y_test)

from sklearn.metrics import mean_squared_error
rfr = RandomForestRegressor(n_estimators=500, random_state=1111)
rfr.fit(X_train, y_train)
test_predictions = rfr.predict(X_test)
mse = mean_squared_error(y_test, test_predictions)