from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Parameters for RandomForestClassifier
# n_estimators = number of trees in the forest
# max_depth = maximum depth of the tree
# random_state = seed for random number generator

rfr = RandomForestRegressor(n_estimators=50, max_depth=10)
rfr.random_state = 1111