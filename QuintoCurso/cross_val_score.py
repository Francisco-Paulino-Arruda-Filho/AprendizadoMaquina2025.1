from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

cross_val_score(estimator=rfc, X=X, cv=5)