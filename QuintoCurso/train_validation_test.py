from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

X_temp, X_validation, y_temp, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=25, random_state=1111, max_depth=5)
rfr.fit(X_train, y_train)
print("Training: {:.2f}".format(rfr.score(X_train, y_train)))