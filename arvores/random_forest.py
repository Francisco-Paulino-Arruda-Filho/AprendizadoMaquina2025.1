from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

rf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
rf.fit(X_train, y_train)    
y_pred = rf.predict(X_test)
print("Predictions:", y_pred)
mse_value = mse(y_test, y_pred)

rf_reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12,random_state=RANDOM_SEED)
rf_reg.fit(X_train, y_train)
y_pred_reg = rf_reg.predict(X_test)
print("Predictions (regression):", y_pred_reg)
rmse_value_reg = mse(y_test, y_pred_reg)
print("RMSE (regression):", rmse_value_reg ** 0.5)

importances = rf_reg.feature_importances_
indices = pd.Series(rf_reg.feature_importances_, index=X.columns)
sorted_importances_rf = indices.sort_values(ascending=False)
plt.figure(figsize=(10, 6))