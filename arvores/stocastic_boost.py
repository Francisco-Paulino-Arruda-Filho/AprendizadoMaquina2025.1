from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

class StochasticBoost:
    def __init__(self, n_estimators=10, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=2, random_state=self.random_state)
            model.fit(X, y)
            self.models.append(model)
            y_pred = model.predict(X)
            residuals = y - y_pred
            X = X + self.learning_rate * residuals.reshape(-1, 1)

    def predict(self, X):
        predictions = sum(model.predict(X) for model in self.models) / len(self.models)
        return predictions
    
X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

sgbt = GradientBoostingRegressor(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=2,
    random_state=RANDOM_SEED
)

sgbt.fit(X_train, y_train)
y_pred = sgbt.predict(X_test)
print("Predictions:", y_pred)
mse_value = mse(y_test, y_pred)
rmse_value = mse_value ** 0.5
print("RMSE:", rmse_value)
