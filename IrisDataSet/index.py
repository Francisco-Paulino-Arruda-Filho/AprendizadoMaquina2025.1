import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Iris.csv")

X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(columns=["Species"]),
    dataset["Species"],
    test_size=0.2,
    random_state=42,
)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
