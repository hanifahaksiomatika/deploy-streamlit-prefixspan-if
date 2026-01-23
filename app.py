import streamlit as st
import pandas as pd

from src.preprocess import clean_and_prepare, build_sequences, build_order_table, build_customer_features
from src.mining import run_prefixspan
from src.anomaly import run_isolation_forest

st.set_page_config(page_title="Dashboard PrefixSpan + Isolation Forest", layout="wide")

st.title("Dashboard Analisis Pola Pembelian & Deteksi Pelanggan Impulsif")
st.caption(
    "Upload data transaksi → filter tahun & umur (range) → PrefixSpan (pola sekuensial) "
    "+ Isolation Forest (deteksi pelanggan impulsif + skor anomali)."
)

with st.sidebar:
    st.header("Input & Parameter")

uploaded = st.sidebar.file_uploader("Upload CSV transaksi", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV transaksi untuk memulai.")
    st.stop()

# Read CSV
try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

# Validate minimal columns early (we'll also alias/rename inside preprocess)
if raw_df.shape[0] == 0:
    st.warning("File CSV kosong.")
    st.stop()

# --- Sidebar filters: Year range and Age range ---
with st.sidebar:
    st.subheader("Filter Tahun Transaksi (Range)")

# Detect available years from order_date (best effort)
try:
    _dates_tmp = pd.to_datetime(raw_df.get("order_date", pd.Series([], dtype="object")), errors="coerce")
    _years = _dates_tmp.dt.year.dropna().astype(int)
    if len(_years) > 0:
        min_year_data = int(_years.min())
        max_year_data = int(_years.max())
    else:
        min_year_data, max_year_data = 2000, 2030
except Exception:
    min_year_data, max_year_data = 2000, 2030

c1, c2 = st.sidebar.columns(2)
year_start = c1.number_input("Dari tahun", min_value=min_year_data, max_value=max_year_data, value=min_year_data, step=1)
year_end   = c2.number_input("Sampai tahun", min_value=min_year_data, max_value=max_year_data, value=max_year_data, step=1)

if year_start > year_end:
    year_start, year_end = year_end, year_start
    st.sidebar.info("Range tahun dibalik otomatis biar valid.")

with st.sidebar:
    st.subheader("Filter Umur Pelanggan (Range)")
    st.caption("Isi sesuai kebutuhan (mis. Gen Z) — default mengikuti min/max umur di data.")

# Detect age range
age_col_guess = None
for cand in ["customer_age", "age", "umur", "usia"]:
    if cand in raw_df.columns:
        age_col_guess = cand
        break

if age_col_guess is None:
    st.sidebar.warning("Kolom umur pelanggan tidak ditemukan (mis. `customer_age`). Filter umur dimatikan.")
    age_min, age_max = None, None
else:
    ages = pd.to_numeric(raw_df[age_col_guess], errors="coerce").dropna()
    if len(ages) > 0:
        min_age_data = int(ages.min())
        max_age_data = int(ages.max())
    else:
        min_age_data, max_age_data = 10, 60

    c3, c4 = st.sidebar.columns(2)
    age_min = c3.number_input("Umur min", min_value=min_age_data, max_value=max_age_data, value=min_age_data, step=1)
    age_max = c4.number_input("Umur max", min_value=min_age_data, max_value=max_age_data, value=max_age_data, step=1)

    if age_min > age_max:
        age_min, age_max = age_max, age_min
        st.sidebar.info("Range umur dibalik otomatis biar valid.")

with st.sidebar:
    st.subheader("PrefixSpan")
    min_support_ratio = st.number_input("Minimum support (ratio)", min_value=0.001, max_value=1.0, value=0.008, step=0.001, format="%.3f")
    min_pattern_len = st.number_input("Minimum panjang pola", min_value=1, max_value=10, value=2, step=1)

    st.subheader("Isolation Forest")
    contamination = st.number_input("Contamination", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=100, step=50)
    score_percentile = st.number_input("Threshold impulsif (persentil skor)", min_value=50, max_value=99, value=95, step=1)

# Run pipeline
with st.spinner("Memproses data..."):
    df = clean_and_prepare(raw_df)

    df_filtered = df.copy()
    # apply year filter (based on parsed datetime)
    df_filtered = df_filtered[df_filtered["order_year"].between(int(year_start), int(year_end))].copy()

    # apply age filter if available
    if age_min is not None and age_max is not None and "customer_age" in df_filtered.columns:
        df_filtered = df_filtered[pd.to_numeric(df_filtered["customer_age"], errors="coerce").between(int(age_min), int(age_max))].copy()

    if df_filtered.empty:
        st.error("Data kosong setelah filter tahun/umur. Coba longgarkan range filternya.")
        st.stop()

    seq_df = build_sequences(df_filtered)
    sequences_all = seq_df["sequence"].tolist()

    order_tbl = build_order_table(df_filtered)
    cust_feat = build_customer_features(order_tbl)

    hasil_if, if_info = run_isolation_forest(
        cust_feat,
        contamination=float(contamination),
        n_estimators=int(n_estimators),
        score_percentile=int(score_percentile),
        random_state=42,
    )

    info_ps, df_pat = run_prefixspan(sequences_all, float(min_support_ratio), min_len=int(min_pattern_len))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "PrefixSpan", "Impulsif (IF)", "Normal vs Impulsif"])

with tab1:
    st.subheader("Ringkasan Data")
    cA, cB, cC = st.columns(3)
    cA.metric("Customer unik", int(df_filtered["customer_id"].nunique()))
    cB.metric("Order unik", int(df_filtered["order_id"].nunique()))
    cC.metric("Kategori unik", int(df_filtered["category"].nunique()))

    st.write("**Periode transaksi (setelah filter):**", f"{int(df_filtered['order_year'].min())} – {int(df_filtered['order_year'].max())}")
    if "customer_age" in df_filtered.columns:
        st.write("**Rentang umur (setelah filter):**", f"{int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').min())} – {int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').max())}")

    st.caption("Preview data setelah pembersihan & filter")
    st.dataframe(df_filtered.head(50), use_container_width=True)

    st.caption("Ringkasan order table & fitur customer")
    c1, c2 = st.columns(2)
    c1.write("Order-level (order_tbl)")
    c1.dataframe(order_tbl.head(20), use_container_width=True)
    c2.write("Customer features (cust_feat)")
    c2.dataframe(cust_feat.head(20), use_container_width=True)

with tab2:
    st.subheader("Hasil PrefixSpan")
    st.dataframe(info_ps, use_container_width=True)
    st.dataframe(df_pat, use_container_width=True, height=480)

    st.download_button(
        "Download pola (CSV)",
        data=df_pat.to_csv(index=False).encode("utf-8"),
        file_name="prefixspan_patterns.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Hasil Isolation Forest (Deteksi Pelanggan Impulsif)")
    st.dataframe(if_info, use_container_width=True)

    st.caption("Daftar customer (urut skor anomali tertinggi)")
    st.dataframe(hasil_if.sort_values("anom_score", ascending=False), use_container_width=True, height=520)

    st.download_button(
        "Download pelanggan impulsif (CSV)",
        data=hasil_if.to_csv(index=False).encode("utf-8"),
        file_name="impulsive_customers_iforest.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Perbandingan Pola Normal vs Impulsif")

    # merge sequences with status
    seq_labeled = seq_df.merge(hasil_if[["customer_id", "status", "anom_score"]], on="customer_id", how="inner")
    normal_seqs = seq_labeled[seq_labeled["status"] == "Normal"]["sequence"].tolist()
    imp_seqs = seq_labeled[seq_labeled["status"] == "Impulsif"]["sequence"].tolist()

    MIN_CUSTOMERS_PATTERN = 20
    colL, colR = st.columns(2)

    with colL:
        st.markdown("### Normal")
        if len(normal_seqs) < MIN_CUSTOMERS_PATTERN:
            st.info(f"Sequence normal terlalu sedikit ({len(normal_seqs)}). Minimal {MIN_CUSTOMERS_PATTERN} untuk mining pola.")
        else:
            info_n, pat_n = run_prefixspan(normal_seqs, float(min_support_ratio), min_len=int(min_pattern_len))
            st.dataframe(info_n, use_container_width=True)
            st.dataframe(pat_n, use_container_width=True, height=420)
            st.download_button(
                "Download pola normal (CSV)",
                data=pat_n.to_csv(index=False).encode("utf-8"),
                file_name="prefixspan_patterns_normal.csv",
                mime="text/csv",
            )

    with colR:
        st.markdown("### Impulsif")
        if len(imp_seqs) < MIN_CUSTOMERS_PATTERN:
            st.info(f"Sequence impulsif terlalu sedikit ({len(imp_seqs)}). Minimal {MIN_CUSTOMERS_PATTERN} untuk mining pola.")
        else:
            info_i, pat_i = run_prefixspan(imp_seqs, float(min_support_ratio), min_len=int(min_pattern_len))
            st.dataframe(info_i, use_container_width=True)
            st.dataframe(pat_i, use_container_width=True, height=420)
            st.download_button(
                "Download pola impulsif (CSV)",
                data=pat_i.to_csv(index=False).encode("utf-8"),
                file_name="prefixspan_patterns_impulsive.csv",
                mime="text/csv",
            )
