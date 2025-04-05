from sklearn.model_selection import GridSearchCV

steps = [
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=6)), # type: ignore
]
pipeline = Pipeline(steps) # type: ignore
parameters = {
    "knn__n_neighbors": np.arange(1, 50),
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1) # type: ignore
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters) # type: ignore
cv.fit(X_train, Y_train)
print(cv.best_score_)
print(cv.best_params_)