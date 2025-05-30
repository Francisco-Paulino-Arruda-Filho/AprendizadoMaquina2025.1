from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score  
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=RANDOM_SEED
)

lr = LogisticRegression(random_state=RANDOM_SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)

classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]

for clf_name in classifiers:
    clf_name.fit(X_train, y_train)

    y_pred = clf_name.pred(X_test)
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
