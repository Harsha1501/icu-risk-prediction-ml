import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.data_loader import preprocess_patient, create_sequences
from src.model import LSTMModel
from src.train import train_model
from src.uncertainty import mc_dropout_predict
from src.evaluate import evaluate

# ==========================
# CONFIGURATION (MATCH DATA)
# ==========================
FEATURES = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
LABEL = "SepsisLabel"

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/raw/Dataset.csv")

X_all, y_all = [], []

for pid, patient_df in df.groupby("Patient_ID"):
    X, y = preprocess_patient(patient_df, FEATURES, LABEL)

    if len(X) < 40:
        continue

    Xs, ys = create_sequences(X, y)
    X_all.append(Xs)
    y_all.append(ys)

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)

print("Final dataset shape:", X_all.shape, y_all.shape)

# ==========================
# TRAIN MODEL
# ==========================
model = LSTMModel(input_dim=len(FEATURES))
train_model(model, X_all, y_all)

# ==========================
# UNCERTAINTY ESTIMATION
# ==========================
mean_pred, uncertainty = mc_dropout_predict(model, X_all)

metrics = evaluate(y_all, mean_pred.squeeze())
print(metrics)

print("Mean uncertainty:", uncertainty.mean())
print("Max uncertainty:", uncertainty.max())
