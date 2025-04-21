from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier()

print(rfc.get_params())

log_reg_clf = LogisticRegression()
#print(log_reg_clf.get_params())