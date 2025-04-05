from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()
linreg_cv = cross_val_score(model, X_train, y_train, cv=kf)

print(np.sqrt(linreg_cv))