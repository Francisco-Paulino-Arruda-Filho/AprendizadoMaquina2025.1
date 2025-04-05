from sklearn.metrics import roc_auc_score
print("AUC:", roc_auc_score(y_test, y_pred_probs))
