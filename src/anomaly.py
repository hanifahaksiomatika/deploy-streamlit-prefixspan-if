import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def run_isolation_forest(
    cust_feat: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    score_percentile: int = 95,
    random_state: int = 42,
):
    """
    Train Isolation Forest on customer features:
    - scale numeric features
    - anom_score = -score_samples (higher = more anomalous)
    - status = Impulsif jika anom_score >= percentile threshold
      (NOTE: ini lebih stabil untuk dashboard daripada label -1, karena user bisa atur persen)
    """
    t0 = time.perf_counter()

    df = cust_feat.copy()
    if df.empty:
        return pd.DataFrame(), {"error": "customer features empty"}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "customer_id"]
    if not num_cols:
        return pd.DataFrame(), {"error": "no numeric features"}

    X = df[num_cols].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination),
        random_state=int(random_state),
    )
    model.fit(Xs)

    # score: higher = more anomalous
    anom_score = -model.score_samples(Xs)

    hasil = df[["customer_id"]].copy()
    hasil[num_cols] = df[num_cols]
    hasil["anom_score"] = anom_score

    # threshold from percentile
    thr = float(np.percentile(anom_score, int(score_percentile)))
    hasil["status"] = np.where(hasil["anom_score"] >= thr, "Impulsif", "Normal")

    t1 = time.perf_counter()

    info = {
        "n_customers": int(len(hasil)),
        "count_norm": int((hasil["status"] == "Normal").sum()),
        "count_imp": int((hasil["status"] == "Impulsif").sum()),
        "score_percentile": int(score_percentile),
        "threshold": float(thr),
        "runtime_sec": float(t1 - t0),
    }
    return hasil.sort_values("anom_score", ascending=False).reset_index(drop=True), info
