from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

dt = dt = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)

ada = AdaBoostClassifier(base_estimator=dt, n_estimators=10, random_state=RANDOM_SEED)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print("Predictions:", y_pred)

y_pred_proba = ada.predict_proba(X_test)[:, 1]
print("Predicted probabilities:", y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC:", roc_auc)
accuracy = accuracy_score(y_test, y_pred)