import streamlit as st
import pandas as pd
import numpy as np

from src.preprocess import (
    clean_and_prepare,
    apply_filters,
    build_order_table,
    build_customer_features,
    build_sequences,)
from src.mining import run_prefixspan
from src.anomaly import run_isolation_forest
from src.interpret import (
    interpret_prefixspan_topk,
    interpret_iforest_topk,
    compare_patterns,)

st.set_page_config(page_title="Dashboard PrefixSpan + Isolation Forest", layout="wide")

st.title("Dashboard Analisis Pola Pembelian Pada Pelanggan Impulsif")
st.caption(
    "Upload data transaksi → filter tahun & umur → Isolation Forest & PrefixSpan")

# Sidebar: Input 
with st.sidebar:
    st.header("Input & Parameter")
    uploaded = st.file_uploader("Upload CSV transaksi", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV transaksi untuk memulai.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

if raw_df.empty:
    st.warning("File CSV kosong.")
    st.stop()

# Preprocess minimal (parse date, columns)
with st.spinner("Menyiapkan data..."):
    df = clean_and_prepare(raw_df)

# Sidebar: Filters 
with st.sidebar:
    st.subheader("Filter Data")

    min_year_data = int(df["order_year"].min()) if "order_year" in df.columns and df["order_year"].notna().any() else 2000
    max_year_data = int(df["order_year"].max()) if "order_year" in df.columns and df["order_year"].notna().any() else 2030

    c1, c2 = st.columns(2)
    year_start = c1.number_input("Tahun mulai", min_value=min_year_data, max_value=max_year_data, value=min_year_data, step=1)
    year_end   = c2.number_input("Tahun akhir", min_value=min_year_data, max_value=max_year_data, value=max_year_data, step=1)
    if year_start > year_end:
        year_start, year_end = year_end, year_start
        st.info("Rentang tahun dibalik otomatis agar valid.")

    age_available = "customer_age" in df.columns
    if age_available:
        ages = pd.to_numeric(df["customer_age"], errors="coerce").dropna()
        if len(ages) > 0:
            min_age_data = int(ages.min())
            max_age_data = int(ages.max())
        else:
            min_age_data, max_age_data = 10, 60
        c3, c4 = st.columns(2)
        age_min = c3.number_input("Umur min", min_value=min_age_data, max_value=max_age_data, value=min_age_data, step=1)
        age_max = c4.number_input("Umur max", min_value=min_age_data, max_value=max_age_data, value=max_age_data, step=1)
        if age_min > age_max:
            age_min, age_max = age_max, age_min
            st.info("Rentang umur dibalik otomatis agar valid.")
    else:
        age_min, age_max = None, None
        st.warning("Kolom `customer_age` tidak ditemukan → filter umur dimatikan.")

    min_seq_len = st.slider("Minimum panjang sequence pelanggan", min_value=1, max_value=10, value=2, step=1)

with st.sidebar:
    st.subheader("PrefixSpan")
    min_support_ratio = st.number_input(
        "Minimum support (ratio)", min_value=0.001, max_value=1.0, value=0.008, step=0.001, format="%.3f")
    min_pattern_len = st.number_input("Minimum panjang pola", min_value=1, max_value=10, value=2, step=1)
    max_pattern_len = st.number_input("Maksimum panjang pola", min_value=2, max_value=10, value=5, step=1)

with st.sidebar:
    st.subheader("Isolation Forest")
    contamination = st.number_input("Contamination", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=100, step=50)
    score_percentile = st.number_input("Threshold impulsif (persentil skor)", min_value=50, max_value=99, value=95, step=1)

with st.spinner("Memproses data & menjalankan model..."):
    df_filtered = apply_filters(
        df,
        year_start=int(year_start),
        year_end=int(year_end),
        age_min=None if age_min is None else int(age_min),
        age_max=None if age_max is None else int(age_max),
        segment=segment,)

    if df_filtered.empty:
        st.error("Data kosong setelah filter")
        st.stop()

    order_tbl = build_order_table(df_filtered)
    cust_feat = build_customer_features(order_tbl)

    hasil_if, if_info = run_isolation_forest(
        cust_feat,
        contamination=float(contamination),
        n_estimators=int(n_estimators),
        score_percentile=int(score_percentile),
        random_state=42,)

    seq_df = build_sequences(df_filtered, hasil_if, min_seq_len=int(min_seq_len))
    sequences_all = seq_df["sequence"].tolist()

    # PrefixSpan overall
    info_ps, df_pat = run_prefixspan(
        sequences_all,
        min_support_ratio=float(min_support_ratio),
        min_len=int(min_pattern_len),
        max_len=int(max_pattern_len),)


# Halaman
st.header("Interpretasi")

cA, cB, cC = st.columns(3)
cA.metric("Jumlah customer", int(df_filtered["customer_id"].nunique()))
cB.metric("Jumlah order", int(df_filtered["order_id"].nunique()))
cC.metric("Jumlah kategori", int(df_filtered["category"].nunique()))

st.write("**Periode transaksi: **", f"{int(df_filtered['order_year'].min())} – {int(df_filtered['order_year'].max())}")
if "customer_age" in df_filtered.columns:
    age_num = pd.to_numeric(df_filtered["customer_age"], errors="coerce").dropna()
    if not age_num.empty:
        st.write("**Rentang umur (setelah filter):**", f"{int(age_num.min())} – {int(age_num.max())}")

total_cust = int(hasil_if["customer_id"].nunique()) if "customer_id" in hasil_if.columns else int(df_filtered["customer_id"].nunique())
n_imp = int((hasil_if["status"] == "Impulsif").sum()) if "status" in hasil_if.columns else 0
pct_imp = (n_imp / total_cust * 100) if total_cust else 0.0

st.subheader("Normal vs Impulsif (Isolation Forest)")
st.write(
    f"Pada dashboard ini, **Impulsif** adalah customer dengan **skor anomali (anom_score) tinggi** menurut Isolation Forest "
    f"(diambil dari **persentil {int(if_info.get('score_percentile', score_percentile))}**), sedangkan **Normal** adalah sisanya. "
    f"Pada data yang dipilih, terdeteksi **{n_imp:,} customer impulsif** dari **{total_cust:,} customer** (**{pct_imp:.2f}%**).")
st.caption("Catatan: label *impulsif* di sini adalah definisi operasional berbasis anomali transaksi, bukan diagnosis psikologis.")

st.subheader("Waktu Proses")
rt_ps = float(info_ps.get("runtime_sec", 0.0)) if isinstance(info_ps, dict) else 0.0
rt_if = float(if_info.get("runtime_sec", 0.0)) if isinstance(if_info, dict) else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("PrefixSpan", f"{rt_ps:.4f} detik")
c2.metric("Isolation Forest", f"{rt_if:.4f} detik")
c3.metric("Total (≈)", f"{(rt_ps + rt_if):.4f} detik")

with st.expander("Detail: apa yang bikin runtime naik/turun?"):
    st.markdown(
        "- **PrefixSpan** paling sensitif terhadap jumlah & panjang sequence, serta parameter `minimum support` dan panjang pola.\n"
        "  Support makin kecil / pola makin panjang → pola makin banyak → waktu biasanya makin lama.\n"
        "- **Isolation Forest** dipengaruhi jumlah customer, jumlah fitur numerik, dan `n_estimators` "
        "(pohon makin banyak → waktu makin lama).\n")

st.subheader("Pengaruh Impulsif")
if_lines = interpret_iforest_topk(hasil_if, top_k=3)
if if_lines:
    for line in if_lines:
        st.markdown(f"- {line}")
else:
    st.info("Belum ada interpretasi impulsif yang bisa ditampilkan")

st.subheader("Pola Pembelian — PrefixSpan")
ins_all, top_all = interpret_prefixspan_topk(df_pat, sequences_all, top_k=3)
if ins_all:
    st.markdown("**Interpretasi Top 3 (berdasarkan support):**")
    for t in ins_all:
        st.markdown(f"- {t}")
    if top_all is not None:
        st.caption("Ringkasan Top 3 pola")
        st.dataframe(top_all, use_container_width=True)
else:
    st.info("Belum ada pola yang cukup untuk diinterpretasikan")

st.subheader("Pola Normal vs Impulsif")

seq_labeled = seq_df.merge(
    hasil_if[["customer_id", "status", "anom_score"]],
    on="customer_id",
    how="inner",
    suffixes=("_seq", "_if"),)

status_col = "status"
if status_col not in seq_labeled.columns:
    for cand in ("status_if", "status_y"):
        if cand in seq_labeled.columns:
            status_col = cand
            break

seq_col = "sequence"
if seq_col not in seq_labeled.columns:
    for cand in ("sequence_seq", "sequence_x"):
        if cand in seq_labeled.columns:
            seq_col = cand
            break

normal_seqs = seq_labeled.loc[seq_labeled[status_col] == "Normal", seq_col].tolist()
imp_seqs = seq_labeled.loc[seq_labeled[status_col] == "Impulsif", seq_col].tolist()

colL, colR = st.columns(2)
pat_n = pd.DataFrame()
pat_i = pd.DataFrame()

MIN_SEQ_FOR_MINING = 20

with colL:
    st.markdown("### Normal")
    if len(normal_seqs) < MIN_SEQ_FOR_MINING:
        st.info(f"Sequence Normal terlalu sedikit ({len(normal_seqs)}). Minimal {MIN_SEQ_FOR_MINING} untuk mining pola.")
    else:
        info_n, pat_n = run_prefixspan(
            normal_seqs,
            min_support_ratio=float(min_support_ratio),
            min_len=int(min_pattern_len),
            max_len=int(max_pattern_len),)
        ins_n, top_n = interpret_prefixspan_topk(pat_n, normal_seqs, top_k=3)
        if ins_n:
            for t in ins_n:
                st.markdown(f"- {t}")
            if top_n is not None:
                st.caption("Ringkasan Top 3 pola Normal")
                st.dataframe(top_n, use_container_width=True)
        else:
            st.info("Pola Normal tidak cukup untuk diinterpretasikan")

with colR:
    st.markdown("### Impulsif")
    if len(imp_seqs) < MIN_SEQ_FOR_MINING:
        st.info(f"Sequence Impulsif terlalu sedikit ({len(imp_seqs)}). Minimal {MIN_SEQ_FOR_MINING} untuk mining pola.")
    else:
        info_i, pat_i = run_prefixspan(
            imp_seqs,
            min_support_ratio=float(min_support_ratio),
            min_len=int(min_pattern_len),
            max_len=int(max_pattern_len),
        )
        ins_i, top_i = interpret_prefixspan_topk(pat_i, imp_seqs, top_k=3)
        if ins_i:
            for t in ins_i:
                st.markdown(f"- {t}")
            if top_i is not None:
                st.caption("Ringkasan Top 3 pola Impulsif")
                st.dataframe(top_i, use_container_width=True)
        else:
            st.info("Pola Impulsif tidak cukup untuk diinterpretasikan")

st.subheader("Interpretasi Perbandingan Normal vs Impulsif")
cmp = compare_patterns(pat_n, pat_i, top_k=5)
if cmp.get("text"):
    for t in cmp["text"]:
        st.markdown(f"- {t}")
if cmp.get("table") is not None and not cmp["table"].empty:
    st.caption("Tabel perbandingan (delta support_ratio)")
    st.dataframe(cmp["table"], use_container_width=True)
