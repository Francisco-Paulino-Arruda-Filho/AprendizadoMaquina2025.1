from sklearn.ensemble import VotingClassifier, KNeighborsClassifier, DecisionTreeClassifier
from sklearn.model_selection import train_test_split

clf_classifier = VotingClassifier(
    estimators=[
        ('Logistic Regression', 'lr'),
        ('K Nearest Neighbours', 'knn'),
        ('Classification Tree', 'dt')
    ],
    voting='hard'
)

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=42
)

clf_classifier.fit(X_train, y_train)
y_pred = clf_classifier.predict(X_test)
print('Voting Classifier Predictions:', y_pred)

knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

# Fit the classifiers
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
# Make predictions
knn_pred = knn.predict(X_test)
dt_pred = dt.predict(X_test)
print('KNN Predictions:', knn_pred)
print('Decision Tree Predictions:', dt_pred)
# Evaluate the classifiers
from sklearn.metrics import accuracy_score
print('KNN Accuracy:', accuracy_score(y_test, knn_pred))
print('Decision Tree Accuracy:', accuracy_score(y_test, dt_pred))