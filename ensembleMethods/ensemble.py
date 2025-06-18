from sklearn.ensemble import MetaEstimator
from sklearn.model_selection import train_test_split

est1 = MetaEstimator(
    estimator='est1',
)

est2 = MetaEstimator(
    estimator='est2',
)

estN = MetaEstimator(
    estimator='estN',
)

est_combined = MetaEstimator(
    estimators=[est1, est2, estN],
    n_estimators=3,
    estimator_type='ensemble',
)

X_train, X_test, y_train, y_test = train_test_split(
    [[0, 0], [1, 1], [1, 0], [0, 1]],
    [0, 1, 1, 0],
    test_size=0.2,
    random_state=42
)

pred = est_combined.predict(x_test)