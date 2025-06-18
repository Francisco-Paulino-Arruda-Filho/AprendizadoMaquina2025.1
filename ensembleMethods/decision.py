# Instantiate the base estimator ("weak" model)
from sklearn.base import accuracy_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


clf_dt = DecisionTreeClassifier(max_depth=3)

# Build the Bagging classifier with 5 estimators
clf_bag = BaggingClassifier(
    clf_dt,
    n_estimators=5
)

# Instantiate the base estimator ("weak" model)
reg_lr = LinearRegression()

# Build the Bagging regressor with 10 estimators
reg_bag = BaggingRegressor(
    reg_lr
)

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression
# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Bagging model to the training set
reg_bag.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_bag.predict(X_test)

clf_bag = BaggingClassifier(
    clf_dt,
    oob_score=True
)
clf_bag.fit(X_train, y_train)

print(clf_bag.oob_score_)

pred = clf_bag.predict(X_test)
print(accuracy_score(y_test, pred))