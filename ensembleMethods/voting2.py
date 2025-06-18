from sklearn.ensemble import VotingClassifier, KNeighborsClassifier, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Instantiate the individual models
clf_knn = KNeighborsClassifier(5)
clf_dt = DecisionTreeClassifier()
clf_lr = LogisticRegression()

# Create an averaging classifier
clf_voting = VotingClassifier(
    estimators=[
        ('knn', clf_knn),
        ('dt', clf_dt),
        ('lr', clf_lr)],
    voting='soft',
    weights=[1, 2, 1]
)

model = DecisionTreeClassifier(
    max_depth=3,
)

model1 = LogisticRegression(
    max_iter=1000,
    C=1.0,
)

model2 = LogisticRegression()