import time
import math
import pandas as pd
from prefixspan import PrefixSpan

def run_prefixspan(sequences, min_support_ratio: float = 0.008, min_len: int = 2):
    """
    Jalankan PrefixSpan seperti notebook:
    - min_support_ratio -> min_count = ceil(ratio * N)
    - output DataFrame pola dengan support_count, support_ratio, length, pattern_str
    """
    t0 = time.time()
    sequences = [s for s in sequences if isinstance(s, list) and len(s) > 0]
    N = len(sequences)
    if N == 0:
        info = pd.DataFrame([{
            "Support (ratio)": min_support_ratio,
            "Min support (count)": 0,
            "Jumlah pola": 0,
            "Waktu (detik)": 0.0
        }])
        return info, pd.DataFrame(columns=["support_count", "pattern", "support_ratio", "length", "pattern_str"])

    min_count = max(1, int(math.ceil(min_support_ratio * N)))

    ps = PrefixSpan(sequences)
    # PrefixSpan in this library returns list of (support, pattern)
    patterns = ps.frequent(min_count)
    df_pat = pd.DataFrame(patterns, columns=["support_count", "pattern"])

    if df_pat.empty:
        df_pat = pd.DataFrame(columns=["support_count", "pattern"])

    df_pat["support_ratio"] = df_pat["support_count"] / N
    df_pat["length"] = df_pat["pattern"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    df_pat = df_pat[df_pat["length"] >= int(min_len)].copy()

    df_pat["pattern_str"] = df_pat["pattern"].apply(lambda x: " → ".join(map(str, x)))
    df_pat = df_pat.sort_values(["support_count", "pattern_str"], ascending=[False, True]).reset_index(drop=True)

    t1 = time.time()
    info = pd.DataFrame([{
        "Support (ratio)": float(min_support_ratio),
        "Min support (count)": int(min_count),
        "Jumlah pola": int(df_pat.shape[0]),
        "Waktu (detik)": round(t1 - t0, 4)
    }])

    return info, df_pat
