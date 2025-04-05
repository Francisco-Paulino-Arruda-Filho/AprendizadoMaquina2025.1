from sklearn.preprocessing import StandardScaler


X = music_df.drop("genre", axis=1).values
Y = music_df["genre"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)    
print(np.mean(X), np.std(X))
print(np.mean(x_train_scaled), np.std(x_train_scaled))

steps = [
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=6)), # type: ignore
]
pipeline = Pipeline(steps) # type: ignore

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
knn_unscaled = KNeighborsClassifier(n_neighbors=6) # type: ignore
print(knn_unscaled.score(X_test, Y_test))