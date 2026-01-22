import io
import pandas as pd
import streamlit as st

from src.preprocess import (
    REQUIRED_COLUMNS,
    normalize_columns,
    validate_columns,
    clean_transactions,
    build_sequences_df,
    build_order_table,
    build_customer_features
)
from src.mining import run_prefixspan, mine_patterns_table
from src.anomaly import fit_isolation_forest, score_isolation_forest


st.set_page_config(page_title="Pola Pembelian & Deteksi Impulsif", layout="wide")

st.title("Dashboard Analisis Pola Pembelian & Deteksi Pelanggan Impulsif")
st.caption(
    "Unggah data transaksi → pembersihan data → (opsional) filter tahun transaksi → (opsional) segmentasi Gen Z → "
    "PrefixSpan (pola sekuensial) + Isolation Forest (deteksi pelanggan impulsif & skor anomali)."
)

with st.sidebar:
    st.header("Input & Parameter")

    uploaded = st.file_uploader("Upload CSV transaksi", type=["csv"])
    st.markdown("**Kolom minimal**:")
    st.code(", ".join(REQUIRED_COLUMNS))

    st.divider()
    st.subheader("Segmentasi")
    use_genz_filter = st.checkbox(
        "Aktifkan filter Gen Z (1997–2012)",
        value=False,
        help="Gen Z ditentukan dari perkiraan tahun lahir = order_year - customer_age (selaras dengan rentang umur per tahun di notebook).",
    )
    st.caption("Default: semua usia. Kalau diaktifkan, hanya Gen Z yang dianalisis.")

    st.divider()
    st.subheader("PrefixSpan")
    min_support_ratio = st.number_input(
        "Minimum support (ratio)",
        min_value=0.0001, max_value=0.5, value=0.008, step=0.0005, format="%.4f"
    )
    min_len_pattern = st.slider("Minimum panjang pola", min_value=1, max_value=6, value=2, step=1)
    st.subheader("Sequence")
    min_seq = st.slider("Min panjang sequence (per customer)", 1, 20, 2)
    max_seq = st.slider("Max panjang sequence (per customer)", 1, 20, 5)

    st.divider()
    st.subheader("Isolation Forest")
    contamination = st.slider("Contamination", min_value=0.001, max_value=0.30, value=0.05, step=0.001, format="%.3f")
    n_estimators = st.selectbox("n_estimators", [100, 200, 300, 500], index=0)
    random_state = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

if not uploaded:
    st.info("Upload CSV dulu ya. Setelah itu hasil PrefixSpan dan deteksi impulsif bakal muncul di bawah.")
    st.stop()

# ---------- Load ----------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

raw_df = load_csv(uploaded.getvalue())

# Auto-rename common aliases (optional)
raw_df, renames = normalize_columns(raw_df)
if renames:
    st.info("Auto-rename kolom terdeteksi: " + ", ".join([f"{k} → {v}" for k,v in renames.items()]))

# ---------- Validate ----------
missing = validate_columns(raw_df)
if missing:
    st.error(
        "CSV kamu belum sesuai skema. Kolom yang kurang:\n\n- " + "\n- ".join(missing) +
        "\n\nSilakan rename kolom atau export ulang sesuai format."
    )
    st.stop()

# ---------- Preprocess ----------
spinner_msg = "Bersihin data" + (" + segmentasi Gen Z..." if use_genz_filter else "...")
# ---------- Year filter UI (default: semua tahun) ----------
available_years = []
try:
    _dates = pd.to_datetime(raw_df["order_date"], errors="coerce")
    available_years = sorted(_dates.dt.year.dropna().astype(int).unique().tolist())
except Exception:
    available_years = []

with st.sidebar:
    st.subheader("Filter Tahun Transaksi")
    use_all_years = st.checkbox("Gunakan semua tahun transaksi", value=True)
    years_selected = None
    if not use_all_years and available_years:
        years_selected = st.multiselect(
            "Pilih tahun transaksi",
            options=available_years,
            default=available_years,
            help="Kalau dikosongkan, sistem akan pakai semua tahun."
        )
        if not years_selected:
            years_selected = None

    if available_years:
        st.caption(f"Tahun terdeteksi di data: {', '.join(map(str, available_years))}")

# ---------- Clean ----------
with st.spinner(spinner_msg):
    df_proc = clean_transactions(raw_df, filter_genz=use_genz_filter, keep_years=years_selected)

mode_label = "Gen Z" if use_genz_filter else "Semua usia"
st.success(f"Data siap diproses ✅  |  Mode: {mode_label}  |  Baris setelah proses: {len(df_proc):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "PrefixSpan", "Impulsif (IF)", "Normal vs Impulsif"])

with tab1:
    st.subheader(f"Ringkasan data (setelah pembersihan | mode: {mode_label})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customer unik", int(df_proc["customer_id"].nunique()))
    c2.metric("Order unik", int(df_proc["order_id"].nunique()))
    c3.metric("Kategori unik", int(df_proc["category"].nunique()))
    st.dataframe(df_proc.head(50), use_container_width=True)

# ---------- Sequence + PrefixSpan ----------
with st.spinner("Membentuk sequence & mining PrefixSpan..."):
    seq_df = build_sequences_df(df_proc, min_seq=min_seq, max_seq=max_seq)
    sequences_all = seq_df["sequence"].tolist()

    info_all, pat_all = run_prefixspan(sequences_all, min_support_ratio=min_support_ratio)
    pat_tbl_all = mine_patterns_table(pat_all, min_len=min_len_pattern)

with tab2:
    st.subheader("Pola sekuensial (PrefixSpan)")
    st.write(info_all)

    st.caption("Disortir berdasarkan support tertinggi. Format pola: `Kategori1 → Kategori2 → ...`")
    st.dataframe(pat_tbl_all, use_container_width=True, height=520)

    csv_bytes = pat_tbl_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download pola (CSV)", data=csv_bytes, file_name="prefixspan_patterns.csv", mime="text/csv")

# ---------- Isolation Forest ----------
with st.spinner("Membangun fitur customer-level & menjalankan Isolation Forest..."):
    order_tbl = build_order_table(df_proc)
    cust_feat, feature_cols = build_customer_features(order_tbl)

    if_model, scaler = fit_isolation_forest(
        cust_feat[feature_cols],
        contamination=contamination,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )

    hasil_if = score_isolation_forest(
        cust_feat,
        feature_cols=feature_cols,
        model=if_model,
        scaler=scaler
    )

with tab3:
    st.subheader("Pelanggan impulsif (Isolation Forest)")
    total = len(hasil_if)
    n_imp = int((hasil_if["label_if"] == -1).sum())
    n_norm = total - n_imp

    c1, c2, c3 = st.columns(3)
    c1.metric("Total customer", total)
    c2.metric("Normal", n_norm)
    c3.metric("Impulsif", n_imp)

    st.caption("`anom_score` makin besar = makin anomalous/impulsif (sesuai definisi di notebook: `-score_samples`).")

    impulsif_tbl = hasil_if[hasil_if["label_if"] == -1].sort_values("anom_score", ascending=False)
    st.dataframe(
        impulsif_tbl[["customer_id", "anom_score"] + feature_cols].head(200),
        use_container_width=True,
        height=520
    )

    csv_bytes = impulsif_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("Download pelanggan impulsif (CSV)", data=csv_bytes, file_name="impulsive_customers.csv", mime="text/csv")

# ---------- Compare patterns Normal vs Impulsif ----------
with tab4:
    st.subheader("Perbandingan pola: Normal vs Impulsif")

    seq_labeled = seq_df.merge(
        hasil_if[["customer_id", "label_if", "anom_score"]],
        on="customer_id",
        how="inner"
    )
    seq_labeled["status"] = seq_labeled["label_if"].map({1: "Normal", -1: "Impulsif"})

    sequences_normal = seq_labeled.loc[seq_labeled["status"] == "Normal", "sequence"].tolist()
    sequences_imp = seq_labeled.loc[seq_labeled["status"] == "Impulsif", "sequence"].tolist()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Normal")
        if len(sequences_normal) < 2:
            st.warning("Sequence normal terlalu sedikit buat mining pola.")
        else:
            info_n, pat_n = run_prefixspan(sequences_normal, min_support_ratio=min_support_ratio)
            st.write(info_n)
            st.dataframe(mine_patterns_table(pat_n, min_len=min_len_pattern).head(200), use_container_width=True, height=420)

    with colB:
        st.markdown("### Impulsif")
        if len(sequences_imp) < 2:
            st.warning("Sequence impulsif terlalu sedikit buat mining pola.")
        else:
            info_i, pat_i = run_prefixspan(sequences_imp, min_support_ratio=min_support_ratio)
            st.write(info_i)
            st.dataframe(mine_patterns_table(pat_i, min_len=min_len_pattern).head(200), use_container_width=True, height=420)

    st.divider()
    st.caption("Tips: Kalau pola impulsif terlalu sedikit, turunin `minimum support` atau naikin `max panjang sequence`.")
