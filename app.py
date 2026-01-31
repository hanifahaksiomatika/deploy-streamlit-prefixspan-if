import streamlit as st
import pandas as pd
import numpy as np

import time
from collections import Counter, defaultdict
import io

from src.preprocess import (
    clean_and_prepare,
    apply_filters,
    build_order_table,
    build_customer_features,
    build_sequences,)
from src.mining import run_prefixspan
from src.anomaly import run_isolation_forest
from src.interpret import interpret_prefixspan_topk, interpret_iforest_topk, compare_patterns

MIN_SUPPORT_RATIO = 0.01
MIN_PATTERN_LEN   = 2
MAX_PATTERN_LEN   = 5

N_ESTIMATORS      = 200
CONTAMINATION     = 0.05
SCORE_PERCENTILE  = 95

TOPK_PATTERNS     = 3
TOPK_COMPARE      = 5


def _humanize_token(tok: str) -> str:
    if not isinstance(tok, str):
        tok = str(tok)
    if tok.endswith("_D"):
        return tok[:-2]
    if tok.endswith("_ND"):
        return tok[:-3]
    return tok

def _humanize_pattern_str(p: str) -> str:
    if not isinstance(p, str):
        p = str(p)
    parts = [x.strip() for x in p.split("→")]
    parts = [_humanize_token(x) for x in parts]
    return " → ".join(parts)

def _humanize_text(text: str, categories: list[str]) -> str:
    out = text
    cats = sorted(set([str(c).strip() for c in categories if pd.notna(c)]), key=len, reverse=True)
    for c in cats:
        out = out.replace(f"{c}_D", f"{c}")
        out = out.replace(f"{c}_ND", f"{c}")
    return out

def _discount_profile_label(cust_if: pd.DataFrame) -> dict:
    out = {"label": "Non-Diskon", "text": [], "table": pd.DataFrame()}
    if cust_if is None or cust_if.empty:
        out["text"].append("Data pelanggan kosong, ringkasan diskon tidak bisa dihitung.")
        return out

    cols = [c for c in ["discount_order_ratio", "avg_discount_rate"] if c in cust_if.columns]
    if not cols:
        out["text"].append("Kolom diskon tidak tersedia (atau selalu 0), sehingga semua transaksi diperlakukan sebagai **Non-Diskon**.")
        return out

    tmp = cust_if.copy()
    any_disc = False
    for c in cols:
        s = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
        if (s > 0).any():
            any_disc = True
            break

    out["label"] = "Diskon" if any_disc else "Non-Diskon"
    if not any_disc:
        out["text"].append("Di data setelah filter, nilai diskon terdeteksi **0** (semua order Non-Diskon).")
        return out

    if "status" in tmp.columns:
        med = tmp.groupby("status")[cols].median(numeric_only=True)

        if ("Impulsif" in med.index) and ("Normal" in med.index):
            med_imp = med.loc["Impulsif"]
            med_norm = med.loc["Normal"]

            diff_ratio = float(med_imp.get("discount_order_ratio", 0) - med_norm.get("discount_order_ratio", 0))
            diff_rate  = float(med_imp.get("avg_discount_rate", 0) - med_norm.get("avg_discount_rate", 0))

            out["table"] = pd.DataFrame({
                "Indikator": ["Rasio order Diskon", "Rata-rata nilai diskon per order"],
                "Median Normal": [float(med_norm.get("discount_order_ratio", 0)), float(med_norm.get("avg_discount_rate", 0))],
                "Median Impulsif": [float(med_imp.get("discount_order_ratio", 0)), float(med_imp.get("avg_discount_rate", 0))],
                "Selisih (Impulsif - Normal)": [diff_ratio, diff_rate],})

            if diff_ratio > 0:
                out["text"].append("Pelanggan **Impulsif** cenderung **lebih sering** melakukan order dengan Diskon dibanding pelanggan **Normal**.")
            elif diff_ratio < 0:
                out["text"].append("Pelanggan **Impulsif** cenderung **lebih jarang** melakukan order dengan Diskon dibanding pelanggan **Normal**.")
    return out



def _order_discount_overview(order_tbl: pd.DataFrame, hasil_if: pd.DataFrame | None = None) -> dict:
    out = {"overall": pd.DataFrame(), "by_status": pd.DataFrame()}
    if order_tbl is None or order_tbl.empty:
        return out

    col = None
    for cand in ["has_discount", "hasDiskon", "is_discount", "discount_flag"]:
        if cand in order_tbl.columns:
            col = cand
            break
    if col is None:
        return out

    s = pd.to_numeric(order_tbl[col], errors="coerce").fillna(0).astype(int)
    cnt = s.value_counts().reindex([0, 1], fill_value=0)
    total = int(cnt.sum()) if int(cnt.sum()) else 1

    out["overall"] = pd.DataFrame({
        "Jenis": ["Non-Diskon", "Diskon"],
        "Jumlah order": [int(cnt.loc[0]), int(cnt.loc[1])],
        "Persentase": [round(int(cnt.loc[0]) / total * 100, 2), round(int(cnt.loc[1]) / total * 100, 2)],})

    if (hasil_if is not None) and ("customer_id" in order_tbl.columns) and ("status" in hasil_if.columns) and ("customer_id" in hasil_if.columns):
        tmp = order_tbl.merge(hasil_if[["customer_id", "status"]], on="customer_id", how="left")
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0).astype(int)
        grp = tmp.groupby("status")[col].mean(numeric_only=True).rename("Rasio order Diskon").reset_index()
        grp["Rasio order Diskon"] = (grp["Rasio order Diskon"] * 100).round(2)
        grp["Rasio order Non-Diskon"] = (100 - grp["Rasio order Diskon"]).round(2)
        out["by_status"] = grp[["status", "Rasio order Diskon", "Rasio order Non-Diskon"]]
    return out



def _trigger_stats(sequences: list[list[str]], suffix: str = "_D", top_k: int = 3):
    occ = Counter()
    occ_next = Counter()
    next_map = defaultdict(Counter)

    for seq in sequences:
        if not isinstance(seq, list):
            continue
        for i, tok in enumerate(seq):
            if isinstance(tok, str) and tok.endswith(suffix):
                occ[tok] += 1
                if i < len(seq) - 1:
                    occ_next[tok] += 1
                    next_map[tok][seq[i + 1]] += 1

    rows = []
    for tok, cnt in occ.most_common(top_k):
        with_next = occ_next.get(tok, 0)
        prob_next = (with_next / cnt) if cnt else 0.0
        if with_next and next_map[tok]:
            nxt, nxt_cnt = next_map[tok].most_common(1)[0]
            prob_nxt = nxt_cnt / with_next
        else:
            nxt, prob_nxt = None, 0.0
        rows.append({
            "trigger": _humanize_token(tok),
            "occurrences": int(cnt),
            "prob_buy_again": float(prob_next),
            "top_next": _humanize_token(nxt) if isinstance(nxt, str) else nxt,
            "prob_top_next_given_next": float(prob_nxt),})
    return pd.DataFrame(rows)

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8-sig")


st.set_page_config(page_title="Dashboard Analisis Pola Belanja Impulsif Gen Z", layout="wide")
st.title("Dashboard Analisis Pola Belanja Impulsif Gen Z")
st.caption("Upload data transaksi → filter tahun & umur → hasil interpretasi Isolation Forest & PrefixSpan.")

@st.cache_data(show_spinner=False)
def _load_and_prepare_csv(file_bytes: bytes):
    t0 = time.perf_counter()
    raw_df = pd.read_csv(io.BytesIO(file_bytes))
    df_prepared = clean_and_prepare(raw_df)
    t1 = time.perf_counter()
    return raw_df, df_prepared, (t1 - t0)

raw_df = None
df_base = None
rt_clean = 0.0

# Sidebar
with st.sidebar:
    st.header("Input Data")
    uploaded = st.file_uploader("Upload CSV transaksi", type=["csv"])

    if uploaded is None:
        st.info("Upload file CSV transaksi untuk memulai.")
    else:
        try:
            file_bytes = uploaded.getvalue()
            raw_df, df_base, rt_clean = _load_and_prepare_csv(file_bytes)
        except Exception as e:
            st.error(f"Gagal membaca/memproses CSV: {e}")
            st.stop()

        if raw_df is None or raw_df.empty:
            st.error("File CSV kosong.")
            st.stop()

        if "order_year" in df_base.columns and df_base["order_year"].notna().any():
            yr = pd.to_numeric(df_base["order_year"], errors="coerce").dropna()
            year_min_data = int(yr.min()) if not yr.empty else 1900
            year_max_data = int(yr.max()) if not yr.empty else 2100
        else:
            year_min_data, year_max_data = 1900, 2100
            st.warning("Kolom tanggal tidak kebaca (order_year kosong/invalid).")

        st.divider()
        st.subheader("Filter Tahun Transaksi")
        c_year1, c_year2 = st.columns(2)
        with c_year1:
            year_start = st.number_input(
                "Dari",
                min_value=year_min_data,
                max_value=year_max_data,
                value=year_min_data,
                step=1,
                key="year_start",)
        with c_year2:
            year_end = st.number_input(
                "Sampai",
                min_value=year_min_data,
                max_value=year_max_data,
                value=year_max_data,
                step=1,
                key="year_end",)

        age_active_sidebar = "customer_age" in df_base.columns and df_base["customer_age"].notna().any()
        st.subheader("Filter Umur Pelanggan")
        if age_active_sidebar:
            ag = pd.to_numeric(df_base["customer_age"], errors="coerce").dropna()
            age_min_data = int(ag.min()) if not ag.empty else 0
            age_max_data = int(ag.max()) if not ag.empty else 120

            c_age1, c_age2 = st.columns(2)
            with c_age1:
                age_min = st.number_input(
                    "Min",
                    min_value=age_min_data,
                    max_value=age_max_data,
                    value=age_min_data,
                    step=1,
                    key="age_min",)
            with c_age2:
                age_max = st.number_input(
                    "Max",
                    min_value=age_min_data,
                    max_value=age_max_data,
                    value=age_max_data,
                    step=1,
                    key="age_max",)
        else:
            st.info("Kolom umur tidak tersedia/valid (customer_age). Filter umur tidak diterapkan.")
            age_min, age_max = None, None

        st.subheader("Minimum Panjang Sequence Pelanggan")
        min_seq_len = st.number_input(
            "Min sequence length",
            min_value=1,
            max_value=50,
            value=2,
            step=1,
            key="min_seq_len",)

        st.divider()
        st.subheader("Parameter Model")
        n_estimators_ui = st.number_input(
            "n_estimators (Isolation Forest)",
            min_value=50,
            max_value=1000,
            value=N_ESTIMATORS,
            step=50,
            key="n_estimators",)
        contamination_ui = st.number_input(
            "contamination (Isolation Forest)",
            min_value=0.001,
            max_value=0.5,
            value=float(CONTAMINATION),
            step=0.001,
            format="%.3f",
            key="contamination",)
        min_support_ui = st.number_input(
            "min_support_ratio (PrefixSpan)",
            min_value=0.001,
            max_value=0.5,
            value=float(MIN_SUPPORT_RATIO),
            step=0.001,
            format="%.3f",
            key="min_support_ratio",)

        st.divider()
        show_normal = st.checkbox("Tampilkan pembanding pelanggan Normal", value=False)

if uploaded is None:
    st.stop()

if year_start > year_end:
    year_start, year_end = year_end, year_start
if (age_min is not None) and (age_max is not None) and (age_min > age_max):
    age_min, age_max = age_max, age_min

t_all0 = time.perf_counter()
with st.spinner("Memproses data..."):
    df = df_base.copy()

    age_active = "customer_age" in df.columns and df["customer_age"].notna().any()
    age_min_use = int(age_min) if age_active else None
    age_max_use = int(age_max) if age_active else None

    t0 = time.perf_counter()
    df_filtered = apply_filters(
        df,
        year_start=int(year_start),
        year_end=int(year_end),
        age_min=age_min_use,
        age_max=age_max_use,
        segment="Semua usia",)
    t1 = time.perf_counter()
    rt_filter = t1 - t0

    if df_filtered.empty:
        st.error("Data kosong setelah filter tahun/umur.")
        st.stop()

    # IF features
    t0 = time.perf_counter()
    order_tbl = build_order_table(df_filtered)
    cust_feat = build_customer_features(order_tbl)
    hasil_if, if_info = run_isolation_forest(
        cust_feat,
        contamination=float(contamination_ui),
        n_estimators=int(n_estimators_ui),
        score_percentile=SCORE_PERCENTILE,
        random_state=42,)
    t1 = time.perf_counter()
    rt_if = t1 - t0

    t0 = time.perf_counter()
    seq_df = build_sequences(df_filtered, hasil_if, min_seq_len=int(min_seq_len))
    sequences_all = seq_df["sequence"].tolist()
    info_all, pat_all = run_prefixspan(
        sequences_all,
        float(min_support_ui),
        min_len=MIN_PATTERN_LEN,
        max_len=MAX_PATTERN_LEN,)

    seq_labeled = seq_df.merge(
        hasil_if[["customer_id", "status", "anom_score"]],
        on="customer_id",
        how="left",
        suffixes=("_seq", "_if"),)

    status_col = "status"
    if status_col not in seq_labeled.columns:
        for cand in ("status_if", "status_y"):
            if cand in seq_labeled.columns:
                status_col = cand
                break

    normal_seqs = seq_labeled.loc[seq_labeled[status_col] == "Normal", "sequence"].tolist()
    imp_seqs    = seq_labeled.loc[seq_labeled[status_col] == "Impulsif", "sequence"].tolist()

    info_i, pat_i = run_prefixspan(
        imp_seqs,
        float(min_support_ui),
        min_len=MIN_PATTERN_LEN,
        max_len=MAX_PATTERN_LEN,) if len(imp_seqs) else ({"N_sequences": 0, "n_patterns": 0}, pd.DataFrame())

    if show_normal and len(normal_seqs):
        info_n, pat_n = run_prefixspan(
            normal_seqs,
            float(min_support_ui),
            min_len=MIN_PATTERN_LEN,
            max_len=MAX_PATTERN_LEN,)
    else:
        info_n, pat_n = ({"N_sequences": len(normal_seqs), "n_patterns": 0}, pd.DataFrame())

    t1 = time.perf_counter()
    rt_ps = t1 - t0

t_all1 = time.perf_counter()
rt_total = t_all1 - t_all0

cats = df_filtered["category"].astype(str).str.strip().unique().tolist()

st.subheader("Ringkasan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah customer", int(df_filtered["customer_id"].nunique()))
c2.metric("Jumlah order", int(df_filtered["order_id"].nunique()))
c3.metric("Jumlah kategori", int(df_filtered["category"].nunique()))
c4.metric("Runtime total", f"{rt_total:.3f} dtk")

st.write("**Periode transaksi:**", f"{int(df_filtered['order_year'].min())} – {int(df_filtered['order_year'].max())}")
if age_active:
    st.write("**Umur:**", f"{int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').min())} – {int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').max())}")
else:
    st.info("Kolom umur tidak tersedia/valid, sehingga filter umur tidak diterapkan.")

st.subheader("Diskon vs Non-Diskon")
disc = _discount_profile_label(hasil_if)

ov = _order_discount_overview(order_tbl, hasil_if)
if isinstance(ov.get("overall"), pd.DataFrame) and not ov["overall"].empty:
    st.markdown("**Proporsi order:**")
    st.dataframe(ov["overall"], use_container_width=True, hide_index=True)

if isinstance(ov.get("by_status"), pd.DataFrame) and not ov["by_status"].empty:
    st.markdown("**Proporsi order per status (Normal vs Impulsif):**")
    st.dataframe(ov["by_status"], use_container_width=True, hide_index=True)

st.subheader("Normal vs Impulsif")

if "status" in hasil_if.columns:
    n_total = int(hasil_if["customer_id"].nunique()) if "customer_id" in hasil_if.columns else len(hasil_if)
    n_imp = int((hasil_if["status"] == "Impulsif").sum())
    n_norm = int((hasil_if["status"] == "Normal").sum())
    st.write(
        f"**Impulsif** = customer dengan skor anomali di atas threshold (persentil **{SCORE_PERCENTILE}**). "
        f"**Normal** = customer lainnya. "
        f"Hasil: **{n_imp} impulsif** dan **{n_norm} normal** dari **{n_total} customer**.")

imp_texts = interpret_iforest_topk(hasil_if, top_k=3)
if imp_texts:
    st.markdown("**Interpretasi Top 3 pelanggan impulsif:**")
    for t in imp_texts:
        st.write("- " + _humanize_text(t, cats))

st.subheader("Pola Pembelian pada Pelanggan Impulsif")

insights_i = []

if pat_i is None or pat_i.empty:
    st.info("Pola impulsif belum terbentuk (sequence impulsif terlalu sedikit atau support terlalu ketat).")
else:
    view_i = pat_i.head(10).copy()
    view_i["pola"] = view_i["pattern_str"].apply(_humanize_pattern_str)
    st.dataframe(view_i[["pola","support_count","support_ratio","length"]], use_container_width=True, hide_index=True)

    insights_i, _ = interpret_prefixspan_topk(pat_i, imp_seqs, top_k=TOPK_PATTERNS)
    st.markdown("**Interpretasi (Top 3 pola impulsif):**")
    for t in insights_i:
        st.write("- " + _humanize_text(t, cats))

    tab_d, tab_nd = st.tabs(["Trigger Diskon", "Trigger Non-Diskon"])

with tab_d:
    trig_d = _trigger_stats(imp_seqs, suffix="_D", top_k=3)
    if trig_d.empty:
        st.info("Tidak ada token Diskon yang cukup sering muncul untuk dirangkum sebagai trigger.")
    else:
        trig_d_view = trig_d.copy()
        trig_d_view["Kategori Pemicu"] = trig_d_view["trigger"].apply(_humanize_token)
        trig_d_view["Jumlah Kasus"] = trig_d_view["occurrences"]
        trig_d_view["P(Beli Lagi)"] = (trig_d_view["prob_buy_again"] * 100).round(2).astype(str) + "%"
        trig_d_view["Next Terbanyak"] = trig_d_view["top_next"].apply(lambda x: _humanize_token(x) if isinstance(x, str) else "-")
        trig_d_view["P(Next Terbanyak)"] = (trig_d_view["prob_top_next_given_next"] * 100).round(2).astype(str) + "%"

        st.markdown("**Diskon sebagai pemicu pembelian lanjutan (Impulsif):**")
        st.dataframe(
            trig_d_view[["Kategori Pemicu", "Jumlah Kasus", "P(Beli Lagi)", "Next Terbanyak", "P(Next Terbanyak)"]],
            use_container_width=True,
            hide_index=True,)

        top_row = trig_d.iloc[0].to_dict()
        trigger = _humanize_token(top_row["trigger"])
        p_again = top_row["prob_buy_again"] * 100
        if top_row["top_next"] is not None:
            nxt = _humanize_token(top_row["top_next"])
            p_nxt = top_row["prob_top_next_given_next"] * 100
            st.write(
                f"**Insight:** Setelah pelanggan impulsif membeli **{trigger}**, sekitar **{p_again:.2f}%** kasus diikuti pembelian berikutnya. "
                f"Pembelian berikutnya paling sering **{nxt}** (~{p_nxt:.2f}%).")
        else:
            st.write(
                f"**Insight:** Setelah pelanggan impulsif membeli **{trigger}**, sekitar **{p_again:.2f}%** kasus diikuti pembelian berikutnya. "
                "(Next item spesifik belum cukup dominan untuk dirangkum.)")

with tab_nd:
    trig_nd = _trigger_stats(imp_seqs, suffix="_ND", top_k=3)
    if trig_nd.empty:
        st.info("Tidak ada token Non-Diskon yang cukup sering muncul untuk dirangkum sebagai trigger.")
    else:
        trig_nd_view = trig_nd.copy()
        trig_nd_view["Kategori Pemicu"] = trig_nd_view["trigger"].apply(_humanize_token)
        trig_nd_view["Jumlah Kasus"] = trig_nd_view["occurrences"]
        trig_nd_view["P(Beli Lagi)"] = (trig_nd_view["prob_buy_again"] * 100).round(2).astype(str) + "%"
        trig_nd_view["Next Terbanyak"] = trig_nd_view["top_next"].apply(lambda x: _humanize_token(x) if isinstance(x, str) else "-")
        trig_nd_view["P(Next Terbanyak)"] = (trig_nd_view["prob_top_next_given_next"] * 100).round(2).astype(str) + "%"

        st.markdown("**Non-Diskon sebagai pemicu pembelian lanjutan (Impulsif):**")
        st.dataframe(
            trig_nd_view[["Kategori Pemicu", "Jumlah Kasus", "P(Beli Lagi)", "Next Terbanyak", "P(Next Terbanyak)"]],
            use_container_width=True,
            hide_index=True,)

        top_row = trig_nd.iloc[0].to_dict()
        trigger = _humanize_token(top_row["trigger"])
        p_again = top_row["prob_buy_again"] * 100
        if top_row["top_next"] is not None:
            nxt = _humanize_token(top_row["top_next"])
            p_nxt = top_row["prob_top_next_given_next"] * 100
            st.write(
                f"**Insight:** Setelah pelanggan impulsif membeli **{trigger}**, sekitar **{p_again:.2f}%** kasus diikuti pembelian berikutnya. "
                f"Pembelian berikutnya paling sering **{nxt}** (~{p_nxt:.2f}%).")
        else:
            st.write(
                f"**Insight:** Setelah pelanggan impulsif membeli **{trigger}**, sekitar **{p_again:.2f}%** kasus diikuti pembelian berikutnya. "
                "(Next item spesifik belum cukup dominan untuk dirangkum.)")

insights_n = []
cmp = {"text": [], "table": pd.DataFrame()}
tbl = pd.DataFrame()

if show_normal:
    st.subheader("Pola Pembelian pada Pelanggan Normal")

    if pat_n is None or pat_n.empty:
        st.info("Pola normal belum terbentuk")
    else:
        view_n = pat_n.head(10).copy()
        view_n["pola"] = view_n["pattern_str"].apply(_humanize_pattern_str)
        st.dataframe(view_n[["pola","support_count","support_ratio","length"]], use_container_width=True, hide_index=True)

        insights_n, _ = interpret_prefixspan_topk(pat_n, normal_seqs, top_k=TOPK_PATTERNS)
        st.markdown("**Interpretasi (Top 3 pola normal):**")
        for t in insights_n:
            st.write("- " + _humanize_text(t, cats))

    st.subheader("Perbandingan Pola Normal vs Impulsif")
    cmp = compare_patterns(pat_n, pat_i, top_k=TOPK_COMPARE)
    for t in cmp.get("text", []):
        st.write("- " + _humanize_text(t, cats))
    tbl = cmp.get("table", pd.DataFrame())
    if isinstance(tbl, pd.DataFrame) and not tbl.empty:
        view = tbl.copy()
        if "pattern_str" in view.columns:
            view["pola"] = view["pattern_str"].apply(_humanize_pattern_str)
        cols = ["pola","support_ratio_normal","support_ratio_imp","delta_support_ratio"]
        cols = [c for c in cols if c in view.columns]
        st.dataframe(view[cols].head(15), use_container_width=True, hide_index=True)

st.subheader("Rekomendasi")

rec = []

if disc.get("label") == "Diskon":
    rec.append("- **Diskon:** fokuskan diskon atau promo pada kategori pemicu (token) yang sering muncul di pelanggan **Impulsif**, lalu follow-up dengan rekomendasi kategori **next** yang paling sering.")
    rec.append("- Terapkan **voucher/kupon lanjutan** setelah transaksi Diskon untuk mendorong pembelian berikutnya sesuai urutan yang terdeteksi.")
else:
    rec.append("- **Diskon:** di data setelah filter tidak terdeteksi transaksi Diskon (semua Non-Diskon), jadi strategi Diskon tidak bisa dievaluasi dari dataset ini.")

rec.append("- **Non-Diskon:** kalau pola Non-Diskon dominan, dorong pembelian lanjutan lewat loyalty points, gratis ongkir, bundling, atau rekomendasi produk pelengkap.")

st.write("\n".join(rec))

st.subheader("Download Output (CSV)")
with st.expander("Download hasil pola & interpretasi"):
    if pat_i is None or pat_i.empty:
        df_pola_imp = pd.DataFrame(columns=["pola","pattern_str","support_count","support_ratio","length"])
    else:
        df_pola_imp = pat_i.copy()
        df_pola_imp["pola"] = df_pola_imp["pattern_str"].apply(_humanize_pattern_str)
        df_pola_imp = df_pola_imp[["pola","pattern_str","support_count","support_ratio","length"]]
    
    if show_normal and pat_n is not None and not pat_n.empty:
        df_pola_norm = pat_n.copy()
        df_pola_norm["pola"] = df_pola_norm["pattern_str"].apply(_humanize_pattern_str)
        df_pola_norm = df_pola_norm[["pola","pattern_str","support_count","support_ratio","length"]]
    else:
        df_pola_norm = pd.DataFrame(columns=["pola","pattern_str","support_count","support_ratio","length"])
    
    if show_normal and isinstance(tbl, pd.DataFrame) and not tbl.empty:
        df_pola_cmp = tbl.copy()
        if "pattern_str" in df_pola_cmp.columns:
            df_pola_cmp["pola"] = df_pola_cmp["pattern_str"].apply(_humanize_pattern_str)
        cols = ["pola","pattern_str","support_ratio_normal","support_ratio_imp","delta_support_ratio"]
        cols = [c for c in cols if c in df_pola_cmp.columns]
        df_pola_cmp = df_pola_cmp[cols]
    else:
        df_pola_cmp = pd.DataFrame(columns=["pola","pattern_str","support_ratio_normal","support_ratio_imp","delta_support_ratio"])

    interp_rows = []
    for i, t in enumerate(imp_texts or [], 1):
        interp_rows.append({"bagian": "Interpretasi IF (Top 3 impulsif)", "rank": i, "teks": _humanize_text(t, cats)})
    for i, t in enumerate(insights_i or [], 1):
        interp_rows.append({"bagian": "Interpretasi pola Impulsif (Top 3)", "rank": i, "teks": _humanize_text(t, cats)})
    if show_normal:
        for i, t in enumerate(insights_n or [], 1):
            interp_rows.append({"bagian": "Interpretasi pola Normal (Top 3)", "rank": i, "teks": _humanize_text(t, cats)})
        for i, t in enumerate((cmp.get("text", []) if isinstance(cmp, dict) else []) or [], 1):
            interp_rows.append({"bagian": "Perbandingan pola (ringkasan)", "rank": i, "teks": _humanize_text(t, cats)})
    df_interpretasi = pd.DataFrame(interp_rows)

    if show_normal:
        cdl1, cdl2, cdl3, cdl4 = st.columns(4)
        with cdl1:
            st.download_button("Pola Impulsif (CSV)", _df_to_csv_bytes(df_pola_imp), "pola_impulsif.csv", "text/csv")
        with cdl2:
            st.download_button("Pola Normal (CSV)", _df_to_csv_bytes(df_pola_norm), "pola_normal.csv", "text/csv")
        with cdl3:
            st.download_button("Perbandingan Pola (CSV)", _df_to_csv_bytes(df_pola_cmp), "perbandingan_pola.csv", "text/csv")
        with cdl4:
            st.download_button("Interpretasi (CSV)", _df_to_csv_bytes(df_interpretasi), "interpretasi_pola.csv", "text/csv")
    else:
        cdl1, cdl2 = st.columns(2)
        with cdl1:
            st.download_button("Pola Impulsif (CSV)", _df_to_csv_bytes(df_pola_imp), "pola_impulsif.csv", "text/csv")
        with cdl2:
            st.download_button("Interpretasi (CSV)", _df_to_csv_bytes(df_interpretasi), "interpretasi_pola.csv", "text/csv")
            