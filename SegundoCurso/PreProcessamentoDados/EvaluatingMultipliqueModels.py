import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X_train = music.drop("genre", axis=1).values
Y_train = music["genre"].values
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(), # type: ignore
    "Decision Tree": DecisionTreeClassifier(), # type: ignore
}

results = []

for model_name, model in models.items():
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring="accuracy")
    results.append(scores)
    print(f"{model_name}: {scores.mean()}")
plt.boxplot(results, labels=models.keys())
plt.show()

for name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    print(f"{name} score: {model.score(X_test_scaled, Y_test)}")