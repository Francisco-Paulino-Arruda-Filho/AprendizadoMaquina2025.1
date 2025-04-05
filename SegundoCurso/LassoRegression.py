from sklearn.linear_model import Lasso

scores = []
for alpha in [0.01, 0.1, 1, 10]:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test))

print(scores)