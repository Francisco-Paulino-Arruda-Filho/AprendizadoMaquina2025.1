from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score

SEED = 42

X_train, y_train, x_test, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=SEED
)

dt = DecisionTreeRegressor(criterion='squared_error', max_depth=2, random_state=SEED)
dt.fit(X_train, y_train)
y_pred = dt.predict(x_test)
print("Predictions:", y_pred)
print("MSE:", mse(y_test, y_pred))

MSE_CV = cross_val_score(dt, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)   
print("MSE CV:", -MSE_CV.mean())
print("MSE CV Std:", MSE_CV.std())