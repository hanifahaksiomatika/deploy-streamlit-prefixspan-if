import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

FEATURES = ["jml_order","total_spend","avg_order","std_order","avg_items","discount_rate","avg_gap_days"]

def run_isolation_forest(
    cust_feat: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 100,
    score_percentile: int = 95,
    random_state: int = 42,
):
    """
    Mengikuti notebook:
    - scaling -> fit IF
    - score = -score_samples (lebih besar = lebih anomali)
    - label_if = predict (1 normal, -1 anomali)
    - status: Impulsif jika label_if == -1
    Tambahan: threshold berdasarkan persentil skor untuk info ringkas.
    """
    t0 = time.time()
    df = cust_feat.copy()

    # ensure all features exist
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0

    X = df[FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if_model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination),
        random_state=int(random_state)
    )
    if_model.fit(X_scaled)

    label = if_model.predict(X_scaled)
    score = -if_model.score_samples(X_scaled)

    hasil = df[["customer_id"] + FEATURES].copy()
    hasil["label_if"] = label
    hasil["anom_score"] = score
    hasil["status"] = np.where(hasil["label_if"] == -1, "Impulsif", "Normal")

    # Info summary
    threshold = float(np.percentile(score, int(score_percentile))) if len(score) else 0.0
    runtime = time.time() - t0

    info = pd.DataFrame([{
        "Total customer": int(hasil.shape[0]),
        "Normal (label IF = 1)": int((hasil["status"] == "Normal").sum()),
        "Impulsif (label IF = -1)": int((hasil["status"] == "Impulsif").sum()),
        "Threshold persentil skor": int(score_percentile),
        "Nilai threshold": round(threshold, 6),
        "Runtime IF (detik)": round(runtime, 4)
    }])

    return hasil.sort_values("anom_score", ascending=False).reset_index(drop=True), info
