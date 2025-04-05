from sklearn.model_selection import RandomizedSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_distributions = {
    "alpha": np.range(0.0001, 1.0, 10.0),
    "solver": ['sag', 'lsqr']
}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_distributions, n_iter=10, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)