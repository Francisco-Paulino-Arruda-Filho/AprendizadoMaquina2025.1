from tpot import TPOTClassifier

# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(
    generations=number_generations,
    population_size=population_size,
    offspring_size=offspring_size,
    scoring=scoring_function,
    verbosity=2,
    random_state=2,
    cv=2
)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))
