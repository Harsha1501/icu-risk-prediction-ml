import torch
import numpy as np

def mc_dropout_predict(model, X, n_samples=30):
    model.train()  # keep dropout ON

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(torch.tensor(X, dtype=torch.float32)).numpy()
            preds.append(p)

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.var(axis=0)

    return mean_pred, uncertainty
