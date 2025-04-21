from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble  import cross_val_score
from sklearn.ensemble import RandomForestRegressor

mae = make_scorer(mean_absolute_error, greater_is_better=False)

rfc = RandomForestRegressor(n_estimators=20, max_depth=5,random_state=1111)

mse = make_scorer(mean_absolute_error, greater_is_better=False)

cv_results = cross_val_score(rfc, X, y, cv=5, scoring=mae)
print(cv_results)