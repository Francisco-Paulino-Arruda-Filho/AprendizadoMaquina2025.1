from sklearn.model_selection import KFold
import numpy as np

X = np.array(range(50))
y = np.array([0] * 20 + [1] * 20)

# n_splits: Number of folds to create
# shuffle: Whether to shuffle the data before splitting into batches
# random_state: Seed for the random number generator (for reproducibility)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = kf.split(X)
for train_index, test_index in splits:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, y_train)
    print(X_test, y_test)
