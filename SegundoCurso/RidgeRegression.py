from sklearn.linear_model import Ridge # type: ignore
from sklearn.datasets import make_regression # type: ignore
scores = []
for alpha in [0.1, 1, 10, 100, 1000]:
    # Generating a regression dataset with noise
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
    
    # Creating and fitting the Ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    
    # Evaluating the model's performance using R^2 score
    score = model.score(X, y)
    scores.append(score)
    print(f"Alpha: {alpha}, R^2 Score: {score:.4f}")