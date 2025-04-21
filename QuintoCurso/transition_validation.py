from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=500, random_state=1111)
rf.fit(X_train, y_train)

cd = pd.read_csv("candy_data.csv")
s1 = cd.sample(60, 1111)
s2 = cd.sample(60, 1112)

print("Training: {:.2f}".format(rf.score(X_train, y_train)))