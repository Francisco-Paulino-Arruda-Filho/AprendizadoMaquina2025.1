from sklearn.linear_model import SGDClassifier

log_reg = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, penalty='l2', random_state=42)

linsvm = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, penalty='l2', random_state=42)