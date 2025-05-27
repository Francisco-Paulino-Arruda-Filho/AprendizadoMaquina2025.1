from sklearn.linear_model import LogisticRegression


lr_L1 = LogisticRegression(solver='liblinear', penalty='l1', C=1000000, fit_intercept=False)
lr_L2 = LogisticRegression()

lr_L1.fit(X, y)
print(lr_L1.coef_)

lr_L2.fit(X, y)
print(lr_L2.coef_)