import ast
import numpy as np
import pandas as pd

def _pattern_to_list(pat):
    """Support berbagai format pattern_str: 'A → B', 'A -> B', "['A','B']" """
    if isinstance(pat, (list, tuple)):
        return list(pat)

    if isinstance(pat, str):
        s = pat.strip()
        if "→" in s:
            return [x.strip() for x in s.split("→") if x.strip()]
        if "->" in s:
            return [x.strip() for x in s.split("->") if x.strip()]
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except Exception:
            pass
        if "," in s:
            return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]
        return [s]

    return [str(pat)]

def _sequence_to_list(seq):
    """Sequence harus list; kalau string, parse."""
    if isinstance(seq, list):
        return seq
    if isinstance(seq, tuple):
        return list(seq)
    if isinstance(seq, str):
        s = seq.strip()
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except Exception:
            pass
        if "→" in s:
            return [x.strip() for x in s.split("→") if x.strip()]
        if "->" in s:
            return [x.strip() for x in s.split("->") if x.strip()]
        if "," in s:
            return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]
        return [s]
    return [str(seq)]

def _is_subsequence(pattern, seq):
    j = 0
    for item in seq:
        if j < len(pattern) and item == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return j == len(pattern)

def _support_count(pattern, sequences):
    return sum(1 for s in sequences if _is_subsequence(pattern, s))

def interpret_prefixspan_topk(df_pat: pd.DataFrame, sequences, top_k: int = 3):
    """
    Interpretasi skripsi-friendly: fokus pada support + confidence sekuensial.
    """
    if df_pat is None or df_pat.empty or not sequences:
        return [], None

    sequences = [_sequence_to_list(s) for s in sequences]
    total_seq = max(len(sequences), 1)

    work = df_pat.copy()
    if "pattern_str" in work.columns:
        work["_pat_list"] = work["pattern_str"].apply(_pattern_to_list)
    elif "pattern" in work.columns:
        work["_pat_list"] = work["pattern"].apply(_pattern_to_list)
    else:
        work["_pat_list"] = work.iloc[:, 0].apply(_pattern_to_list)

    if "length" in work.columns:
        work = work[work["length"] >= 2].copy()
    else:
        work = work[work["_pat_list"].apply(lambda x: len(x) >= 2)].copy()

    if work.empty:
        return [], None

    if "support_count" in work.columns:
        top = work.sort_values("support_count", ascending=False).head(top_k)
    else:
        top = work.head(top_k)

    def support_label(supp_ratio: float) -> str:
        if supp_ratio >= 0.05:
            return "sangat sering"
        if supp_ratio >= 0.02:
            return "cukup sering"
        if supp_ratio >= 0.01:
            return "sedang"
        return "jarang"

    insights = []
    for _, row in top.iterrows():
        pat = row["_pat_list"]
        prefix = pat[:-1]
        last = pat[-1]

        supp_cnt = int(row["support_count"]) if "support_count" in row else _support_count(pat, sequences)
        supp_ratio = supp_cnt / total_seq
        prefix_cnt = _support_count(prefix, sequences)
        conf = (supp_cnt / prefix_cnt) if prefix_cnt > 0 else 0.0

        label = support_label(supp_ratio)
        pat_str = " → ".join(pat)
        prefix_str = " → ".join(prefix)

        insights.append(
            f"Pola **{pat_str}** termasuk pola yang **{label}** muncul (support **{supp_ratio*100:.2f}%** "
            f"atau **{supp_cnt}** dari **{total_seq}** sequence). "
            f"Jika pelanggan sudah memiliki urutan **{prefix_str}**, sekitar **{conf*100:.2f}%** "
            f"transaksi berikutnya diikuti oleh **{last}**. "
            f"Insight ini dapat dimanfaatkan sebagai dasar **rekomendasi/cross-sell berbasis urutan**."
        )

    top_view = top.copy()
    top_view["pattern_str"] = top_view["_pat_list"].apply(lambda x: " → ".join(x))
    cols_show = [c for c in ["pattern_str", "support_count", "support_ratio", "length"] if c in top_view.columns]
    top_view = top_view[cols_show] if cols_show else top_view[["pattern_str"]]
    return insights, top_view

def interpret_iforest_topk(hasil_if: pd.DataFrame, top_k: int = 3):
    """
    Interpretasi top-K impulsif: tampilkan skor + 2 fitur numerik yang paling menonjol vs median Normal.
    """
    if hasil_if is None or hasil_if.empty or "anom_score" not in hasil_if.columns:
        return []

    top_imp = hasil_if.sort_values("anom_score", ascending=False).head(top_k)

    num_cols = hasil_if.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "anom_score"]

    baseline = None
    if "status" in hasil_if.columns and num_cols:
        normal = hasil_if[hasil_if["status"] == "Normal"]
        baseline = (normal[num_cols].median(numeric_only=True) if not normal.empty
                    else hasil_if[num_cols].median(numeric_only=True))

    lines = []
    for _, r in top_imp.iterrows():
        cid = r.get("customer_id", "-")
        score = float(r["anom_score"])

        highlight_txt = ""
        if baseline is not None and num_cols:
            ratios = []
            for c in num_cols:
                b = baseline.get(c, np.nan)
                v = r.get(c, np.nan)
                if pd.isna(b) or pd.isna(v):
                    continue
                if b == 0:
                    # hindari inf, ubah jadi label tekstual
                    ratio = None
                else:
                    ratio = v / b
                ratios.append((c, ratio, v, b))

            # urutkan yang ratio terbesar (yang ratio None kita taruh belakang)
            ratios = sorted(ratios, key=lambda x: (-1 if x[1] is None else x[1]), reverse=True)[:2]
            parts = []
            for c, ratio, v, b in ratios:
                if ratio is None:
                    parts.append(f"**{c}** jauh di atas median normal")
                else:
                    parts.append(f"**{c}** ~{ratio:.2f}x median normal")
            if parts:
                highlight_txt = " Indikator menonjol: " + ", ".join(parts) + "."

        lines.append(f"Customer **{cid}** terdeteksi impulsif dengan skor anomali **{score:.4f}**.{highlight_txt}")

    return lines

def compare_patterns(pat_normal: pd.DataFrame, pat_imp: pd.DataFrame, top_k: int = 5):
    """
    Buat tabel delta support_ratio & ringkasan teks:
    - pola lebih dominan di Impulsif
    - pola lebih dominan di Normal
    """
    if (pat_normal is None or pat_normal.empty) and (pat_imp is None or pat_imp.empty):
        return {"text": [], "table": pd.DataFrame()}

    def prep(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=["pattern_str","support_ratio","support_count"])
        cols = ["pattern_str", "support_ratio", "support_count"]
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[cols]

    n = prep(pat_normal).rename(columns={"support_ratio":"support_ratio_normal","support_count":"support_count_normal"})
    i = prep(pat_imp).rename(columns={"support_ratio":"support_ratio_imp","support_count":"support_count_imp"})

    m = n.merge(i, on="pattern_str", how="outer").fillna(0)
    m["delta_support_ratio"] = m["support_ratio_imp"] - m["support_ratio_normal"]

    m = m.sort_values("delta_support_ratio", ascending=False).reset_index(drop=True)

    top_imp = m.head(top_k)
    top_norm = m.tail(top_k).sort_values("delta_support_ratio")

    text = []
    if not top_imp.empty:
        items = ", ".join(top_imp["pattern_str"].head(top_k).tolist())
        text.append(f"Pola yang cenderung lebih dominan pada **Impulsif** (delta support_ratio tertinggi): {items}.")
    if not top_norm.empty:
        items = ", ".join(top_norm["pattern_str"].head(top_k).tolist())
        text.append(f"Pola yang cenderung lebih dominan pada **Normal** (delta support_ratio terendah): {items}.")

    return {"text": text, "table": m}
