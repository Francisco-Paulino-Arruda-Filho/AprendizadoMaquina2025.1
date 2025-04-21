from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV()

param_list = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [2, 4, 6, 8],
    'max_features': [2, 4, 6, 8, 10]
}
# estimator: the model to use
# scoring: the scoring method to use
# n_iter: the number of iterations to perform
# param_distributions: the parameter distributions to sample from   
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

rfc = RandomForestRegressor(n_estimators=100, random_state=1111)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

readom_search = RandomizedSearchCV(
    estimator=rfc,
    param_distributions=param_list,
    scoring=scorer,
    n_iter=10,
    cv=5,
    verbose=1
)

readom_search.fit(X, y)