import streamlit as st
import pandas as pd
import numpy as np

from src.preprocess import (clean_and_prepare,apply_filters,build_order_table,build_customer_features,build_sequences,)
from src.mining import run_prefixspan
from src.anomaly import run_isolation_forest
from src.interpret import (interpret_prefixspan_topk,interpret_iforest_topk,compare_patterns,)

st.set_page_config(page_title="Dashboard PrefixSpan + Isolation Forest", layout="wide")

st.title("Dashboard Analisis Pola Pembelian & Deteksi Pelanggan Impulsif")
st.caption(
    "Upload data transaksi → filter tahun & umur (opsional segmen Gen Z) → "
    "PrefixSpan (pola sekuensial) & Isolation Forest (deteksi pelanggan impulsif + skor anomali).")

# Sidebar: Input 
with st.sidebar:
    st.header("Input & Parameter")
    uploaded = st.file_uploader("Upload CSV transaksi", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV transaksi untuk memulai.")
    st.stop()

# Load CSV 
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

    # Segment: All ages or Gen Z (not mandatory)
    segment = st.selectbox("Segmen", ["Semua usia", "Gen Z (aturan tahun transaksi)"], index=0)

    # Range tahun transaksi dari data
    min_year_data = int(df["order_year"].min()) if "order_year" in df.columns and df["order_year"].notna().any() else 2000
    max_year_data = int(df["order_year"].max()) if "order_year" in df.columns and df["order_year"].notna().any() else 2030

    c1, c2 = st.columns(2)
    year_start = c1.number_input("Tahun mulai", min_value=min_year_data, max_value=max_year_data, value=min_year_data, step=1)
    year_end   = c2.number_input("Tahun akhir", min_value=min_year_data, max_value=max_year_data, value=max_year_data, step=1)
    if year_start > year_end:
        year_start, year_end = year_end, year_start
        st.info("Rentang tahun dibalik otomatis agar valid.")

    # Range umur (custom)
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
        "Minimum support (ratio)", min_value=0.001, max_value=1.0, value=0.008, step=0.001, format="%.3f"
    )
    min_pattern_len = st.number_input("Minimum panjang pola", min_value=1, max_value=10, value=2, step=1)
    max_pattern_len = st.number_input("Maksimum panjang pola", min_value=2, max_value=10, value=5, step=1)

with st.sidebar:
    st.subheader("Isolation Forest")
    contamination = st.number_input("Contamination", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=100, step=50)
    score_percentile = st.number_input("Threshold impulsif (persentil skor)", min_value=50, max_value=99, value=95, step=1)

# Run pipeline 
with st.spinner("Memproses data & menjalankan model..."):
    df_filtered = apply_filters(
        df,
        year_start=int(year_start),
        year_end=int(year_end),
        age_min=None if age_min is None else int(age_min),
        age_max=None if age_max is None else int(age_max),
        segment=segment,)

    if df_filtered.empty:
        st.error("Data kosong setelah filter. Coba longgarkan filter tahun/umur atau nonaktifkan segmen Gen Z.")
        st.stop()

    # order table & features
    order_tbl = build_order_table(df_filtered)
    cust_feat = build_customer_features(order_tbl)

    hasil_if, if_info = run_isolation_forest(
        cust_feat,
        contamination=float(contamination),
        n_estimators=int(n_estimators),
        score_percentile=int(score_percentile),
        random_state=42,)

    # sequences for PrefixSpan include discount token
    seq_df = build_sequences(df_filtered, hasil_if, min_seq_len=int(min_seq_len))
    sequences_all = seq_df["sequence"].tolist()

    # PrefixSpan overall
    info_ps, df_pat = run_prefixspan(
        sequences_all,
        min_support_ratio=float(min_support_ratio),
        min_len=int(min_pattern_len),
        max_len=int(max_pattern_len),)

# Tabs 
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "PrefixSpan", "Impulsif (IF)", "Normal vs Impulsif"])

# Tab 1: Summary
with tab1:
    st.subheader("Ringkasan Data")

    cA, cB, cC = st.columns(3)
    cA.metric("Customer unik", int(df_filtered["customer_id"].nunique()))
    cB.metric("Order unik", int(df_filtered["order_id"].nunique()))
    cC.metric("Kategori unik", int(df_filtered["category"].nunique()))

    st.write("**Periode transaksi (setelah filter):**", f"{int(df_filtered['order_year'].min())} – {int(df_filtered['order_year'].max())}")
    if "customer_age" in df_filtered.columns:
        age_num = pd.to_numeric(df_filtered["customer_age"], errors="coerce").dropna()
        if not age_num.empty:
            st.write("**Rentang umur (setelah filter):**", f"{int(age_num.min())} – {int(age_num.max())}")

    # Interpretasi singkat: proporsi impulsif
    total_cust = int(hasil_if["customer_id"].nunique()) if "customer_id" in hasil_if.columns else int(df_filtered["customer_id"].nunique())
    n_imp = int((hasil_if["status"] == "Impulsif").sum()) if "status" in hasil_if.columns else 0
    pct_imp = (n_imp / total_cust * 100) if total_cust else 0.0
    st.subheader("Interpretasi Singkat")
    st.write(
        f"Data yang dianalisis berjumlah **{len(df_filtered):,}** baris transaksi dengan **{total_cust:,}** pelanggan unik. "
        f"Berdasarkan Isolation Forest, terdeteksi **{n_imp:,}** pelanggan impulsif (**{pct_imp:.2f}%** dari total pelanggan) "
        f"pada rentang data yang dipilih.")

    st.caption("Preview data setelah pembersihan & filter")
    st.dataframe(df_filtered.head(50), use_container_width=True)

    st.caption("Ringkasan order table & fitur customer")
    c1, c2 = st.columns(2)
    c1.write("Order-level (order_tbl)")
    c1.dataframe(order_tbl.head(20), use_container_width=True)
    c2.write("Customer features (cust_feat)")
    c2.dataframe(cust_feat.head(20), use_container_width=True)

# Tab 2: PrefixSpan 
with tab2:
    st.subheader("Hasil PrefixSpan")
    st.dataframe(pd.DataFrame([info_ps]), use_container_width=True)
    st.dataframe(df_pat, use_container_width=True, height=480)

    st.subheader("Interpretasi (Top 3)")
    insights, top_view = interpret_prefixspan_topk(df_pat, sequences_all, top_k=3)
    if not insights:
        st.info("Belum ada pola yang cukup untuk diinterpretasikan. Coba turunkan minimum support atau longgarkan filter.")
    else:
        for t in insights:
            st.markdown(f"- {t}")
        st.caption("Ringkasan Top 3 pola")
        st.dataframe(top_view, use_container_width=True)

        st.download_button(
            "Download interpretasi (TXT)",
            data="\n".join(insights).encode("utf-8"),
            file_name="interpretasi_prefixspan_top3.txt",
            mime="text/plain",)

    st.download_button(
        "Download pola (CSV)",
        data=df_pat.to_csv(index=False).encode("utf-8"),
        file_name="prefixspan_patterns.csv",
        mime="text/csv",)

# Tab 3: Isolation Forest
with tab3:
    st.subheader("Hasil Isolation Forest (Deteksi Pelanggan Impulsif)")
    st.dataframe(pd.DataFrame([if_info]), use_container_width=True)

    st.subheader("Interpretasi (Top 3 pelanggan impulsif)")
    lines = interpret_iforest_topk(hasil_if, top_k=3)
    if lines:
        for t in lines:
            st.markdown(f"- {t}")
    else:
        st.caption("Interpretasi Top 3 tidak tersedia (fitur numerik tidak cukup atau data kosong).")

    st.caption("Daftar customer (urut skor anomali tertinggi)")
    st.dataframe(hasil_if.sort_values("anom_score", ascending=False), use_container_width=True, height=520)

    st.download_button(
        "Download pelanggan impulsif (CSV)",
        data=hasil_if.to_csv(index=False).encode("utf-8"),
        file_name="impulsive_customers_iforest.csv",
        mime="text/csv",)

# Tab 4: Normal vs Impulsif 
with tab4:
    st.subheader("Perbandingan Pola Normal vs Impulsif")

    seq_labeled = seq_df.merge(hasil_if[["customer_id", "status", "anom_score"]], on="customer_id", how="inner")
    normal_seqs = seq_labeled[seq_labeled["status"] == "Normal"]["sequence"].tolist()
    imp_seqs = seq_labeled[seq_labeled["status"] == "Impulsif"]["sequence"].tolist()

    st.caption(f"Jumlah sequence: Normal={len(normal_seqs)} | Impulsif={len(imp_seqs)} (setelah min sequence length = {int(min_seq_len)})")

    MIN_CUSTOMERS_PATTERN = 20
    colL, colR = st.columns(2)

    pat_n = pd.DataFrame()
    pat_i = pd.DataFrame()

    with colL:
        st.markdown("### Normal")
        if len(normal_seqs) < MIN_CUSTOMERS_PATTERN:
            st.info(f"Sequence normal terlalu sedikit ({len(normal_seqs)}). Minimal {MIN_CUSTOMERS_PATTERN} untuk mining pola.")
        else:
            info_n, pat_n = run_prefixspan(normal_seqs, float(min_support_ratio), min_len=int(min_pattern_len), max_len=int(max_pattern_len))
            st.dataframe(pd.DataFrame([info_n]), use_container_width=True)
            st.dataframe(pat_n, use_container_width=True, height=420)

    with colR:
        st.markdown("### Impulsif")
        if len(imp_seqs) < MIN_CUSTOMERS_PATTERN:
            st.info(f"Sequence impulsif terlalu sedikit ({len(imp_seqs)}). Minimal {MIN_CUSTOMERS_PATTERN} untuk mining pola.")
        else:
            info_i, pat_i = run_prefixspan(imp_seqs, float(min_support_ratio), min_len=int(min_pattern_len), max_len=int(max_pattern_len))
            st.dataframe(pd.DataFrame([info_i]), use_container_width=True)
            st.dataframe(pat_i, use_container_width=True, height=420)

    st.subheader("Interpretasi Perbandingan")
    if pat_n.empty and pat_i.empty:
        st.info("Belum ada pola pada kedua kelompok. Coba turunkan minimum support atau turunkan minimum panjang sequence.")
    else:
        summary = compare_patterns(pat_n, pat_i, top_k=5)
        if summary["text"]:
            for t in summary["text"]:
                st.markdown(f"- {t}")

        if summary["table"] is not None and not summary["table"].empty:
            st.caption("Tabel perbandingan pola (delta support_ratio)")
            st.dataframe(summary["table"], use_container_width=True, height=520)
