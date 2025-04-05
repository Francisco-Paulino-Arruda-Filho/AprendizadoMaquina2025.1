# Classificação binária com Regressão Logística
# p > 0.5 -> 1
# p < 0.5 -> 0

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # type: ignore
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]  # Probabilidades de classe 1
print(y_pred_proba[0])