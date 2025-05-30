from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection_ import train_test_split, GridSearchCV

RANDOM_SEED = 42

dt = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=RANDOM_SEED)
dt.get_params()

param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}

grid_dt = GridSearchCV(
    dt,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

grid_dt.fit(X_train, y_train)

best_CV_params = grid_dt.best_params_
best_CV_scores = grid_dt.best_scores_
best_model = grid_dt.best_estimator_
