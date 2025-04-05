from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

music_df = music_df.dropna(subset=["genre", "popularity", "loudess", "liveness","tempo"])
music_df = music["genre"] = np.where(music_df["genre"] == "pop", 1, 0)  

steps = [
    (("imputation"), SimpleImputer(strategy="most_frequent")),
    ("logistic_regression", LogisticRegression(max_iter=1000)),
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)