from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def fit_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
):
    """
    Fit scaler + IsolationForest on customer-level features.
    Returns fitted (model, scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(X_scaled)
    return model, scaler

def score_isolation_forest(
    cust_feat: pd.DataFrame,
    feature_cols: list[str],
    model: IsolationForest,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Score customers:
    - label_if: 1 normal, -1 anomaly
    - anom_score: larger = more anomalous (defined as -score_samples like notebook)
    """
    X_scaled = scaler.transform(cust_feat[feature_cols])
    label = model.predict(X_scaled)
    score = -model.score_samples(X_scaled)

    out = cust_feat.copy()
    out["label_if"] = label
    out["anom_score"] = score
    out["status"] = np.where(out["label_if"] == -1, "Impulsif (Anomali)", "Normal")
    return out
