from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

dt = DecisionTreeRegressor(criterion='mse', max_depth=2, random_state=RANDOM_SEED)
gb = GradientBoostingRegressor(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=2,
    random_state=RANDOM_SEED
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print("Predictions:", y_pred)
mse_value = mse(y_test, y_pred)
rmse_value = mse_value ** 0.5
print("RMSE:", rmse_value)