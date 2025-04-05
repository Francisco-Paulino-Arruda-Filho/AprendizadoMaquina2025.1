from statistics import LinearRegression
from sklearn.model_selection import cross_val_score, KFold # type: ignore
kf = KFold(n_splits=5, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_results = -cv_results  # Convert to positive MSE
print("Cross-validated MSE:", cv_results.mean())
print(np.mean(cv_results))
print(np.std(cv_results))
print(np.quantile(cv_results, [0.25, 0.75]))