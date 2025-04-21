import pandas as pd
from sklearn.model_selection import train_test_split

tic_tac_toe = pd.read_csv('https://raw.githubusercontent.com/renatopp/tic-tac-toe/master/tic-tac-toe.csv')
X = pd.get_dummies(tic_tac_toe.iloc[:, :-9])
y = tic_tac_toe.iloc[:, -9:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)