from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

RANDOM_SEED = 42

rf  = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=RANDOM_SEED)
rf.get_params()
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'min_samples_leaf': [0.01, 0.05, 0.1, 0.12],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['log2', 'sqrt']
}


grid_rf = GridSearchCV(
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    cv=3
)

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_

best_model = grid_rf.best_estimator_

y_pred = best_model.predict(X_test)

rmse_test = y_pred ** 0.5

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))