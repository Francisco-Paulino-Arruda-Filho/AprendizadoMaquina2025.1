from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.ensemble import BaggingClassifier

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
bagging = BaggingClassifier(base_estimator=dt, n_estimators=10, random_state=RANDOM_SEED)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Predictions:", y_pred)