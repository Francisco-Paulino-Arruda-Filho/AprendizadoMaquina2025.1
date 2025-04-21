from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

depth = [4, 6, 8, 10, 12]
sample = [2, 4, 6, 8]
features = [2, 4, 6, 8, 10]
rfc = RandomForestRegressor(
    n_estimators=100,
    max_depth=depth[0],
    min_samples_split=sample[3],
    max_features=features[1]
)

rfc.get_params(
        
)