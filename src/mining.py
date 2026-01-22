from __future__ import annotations

import math
import time
import pandas as pd
from prefixspan import PrefixSpan

def run_prefixspan(seqs: list[list], min_support_ratio: float):
    """
    Ported from your notebook.
    Returns:
      info_df: 1-row DataFrame about runtime/support
      patterns_df: frequent patterns with support + pretty string
    """
    if not seqs:
        info = pd.DataFrame([{
            "Support (ratio)": min_support_ratio,
            "Min support (count)": 0,
            "Jumlah pola": 0,
            "Waktu (detik)": 0.0
        }])
        return info, pd.DataFrame(columns=["support_count","pattern","support_ratio","length","pattern_str"])

    N = len(seqs)
    min_count = math.ceil(min_support_ratio * N)

    ps = PrefixSpan(seqs)

    t0 = time.perf_counter()
    patterns = ps.frequent(min_count)  # list[tuple[support_count, pattern_list]]
    t1 = time.perf_counter()

    df_pat = pd.DataFrame(patterns, columns=["support_count", "pattern"])
    df_pat["support_ratio"] = df_pat["support_count"] / N
    df_pat["length"] = df_pat["pattern"].apply(len)
    df_pat["pattern_str"] = df_pat["pattern"].apply(lambda x: " → ".join(map(str, x)))
    df_pat = df_pat.sort_values(["support_count","pattern_str"], ascending=[False, True]).reset_index(drop=True)

    info = pd.DataFrame([{
        "Support (ratio)": float(min_support_ratio),
        "Min support (count)": int(min_count),
        "Jumlah pola": int(df_pat.shape[0]),
        "Waktu (detik)": round(t1 - t0, 4)
    }])

    return info, df_pat

def mine_patterns_table(df_pat: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    if df_pat.empty:
        return pd.DataFrame(columns=["Pola Pembelian","support_count","support_ratio","length"])

    out = df_pat[df_pat["length"] >= min_len].copy()
    out = out.rename(columns={"pattern_str": "Pola Pembelian"})
    return out[["Pola Pembelian","support_count","support_ratio","length"]]
