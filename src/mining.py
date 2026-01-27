import time
import math
import pandas as pd
from prefixspan import PrefixSpan

def run_prefixspan(seqs, min_support_ratio=0.01, min_len=2, max_len=5):
    """
    PrefixSpan frequent patterns:
    - min_support_ratio -> min_count = ceil(ratio * N), minimal 1
    - filter pattern length
    - output: info dict + DataFrame patterns
    """
    N = len(seqs)
    if N == 0:
        return {"error": "No sequences", "N_sequences": 0}, pd.DataFrame(columns=["support_count","pattern","support_ratio","length","pattern_str"])

    min_count = max(1, int(math.ceil(min_support_ratio * N)))

    ps = PrefixSpan(seqs)
    t0 = time.perf_counter()
    patterns = ps.frequent(min_count)
    t1 = time.perf_counter()

    pat = pd.DataFrame(patterns, columns=["support_count", "pattern"])
    pat["support_ratio"] = pat["support_count"] / N
    pat["length"] = pat["pattern"].apply(len)

    pat = pat[(pat["length"] >= int(min_len)) & (pat["length"] <= int(max_len))].copy()
    pat["pattern_str"] = pat["pattern"].apply(lambda x: " → ".join(map(str, x)))
    pat = pat.sort_values(["support_count", "support_ratio", "length"], ascending=[False, False, True]).reset_index(drop=True)

    info = {
        "min_support_ratio": float(min_support_ratio),
        "min_support_count": int(min_count),
        "N_sequences": int(N),
        "n_patterns": int(len(pat)),
        "runtime_sec": float(t1 - t0),
    }
    return info, pat
