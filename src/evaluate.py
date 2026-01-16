from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(y_true, y_pred):
    return {
        "AUROC": roc_auc_score(y_true, y_pred),
        "AUPRC": average_precision_score(y_true, y_pred)
    }
