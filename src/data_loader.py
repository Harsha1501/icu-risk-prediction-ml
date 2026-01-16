import numpy as np
import pandas as pd

def preprocess_patient(df, feature_cols, label_col):
    # sort by time
    df = df.sort_values("Hour")

    # select features + label
    df = df[feature_cols + [label_col]]

    # handle missing values
    df = df.ffill().fillna(0)

    X = df[feature_cols].values
    y = df[label_col].values

    return X, y


def create_sequences(X, y, seq_len=24, pred_horizon=6):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_len - pred_horizon):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(int(y[i + seq_len:i + seq_len + pred_horizon].max() > 0))

    return np.array(X_seq), np.array(y_seq)
