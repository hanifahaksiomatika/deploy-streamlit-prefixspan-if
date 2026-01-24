import streamlit as st
import pandas as pd

from src.preprocess import clean_and_prepare, build_sequences, build_order_table, build_customer_features
from src.mining import run_prefixspan
from src.anomaly import run_isolation_forest

# ===== Tambahan: Interpretasi otomatis (PrefixSpan & Isolation Forest) =====
import ast
import math

def _pattern_to_list(pat):
    """Pattern bisa list/tuple, atau string 'A -> B', atau string repr list."""
    if isinstance(pat, (list, tuple)):
        return list(pat)
    if isinstance(pat, str):
        if "->" in pat:
            return [x.strip() for x in pat.split("->")]
        try:
            v = ast.literal_eval(pat)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            return [pat]
    return [str(pat)]

def _is_subsequence(pattern, seq):
    """Cek apakah pattern muncul sebagai subsequence (urutan) di seq."""
    j = 0
    for item in seq:
        if j < len(pattern) and item == pattern[j]:
            j += 1
            if j == len(pattern):
                return True
    return j == len(pattern)

def _support_count(pattern, sequences):
    return sum(1 for s in sequences if _is_subsequence(pattern, s))

def build_prefixspan_interpretation(df_pat, sequences, top_k=3):
    """
    Interpretasi Top-K pola PrefixSpan berbasis:
    - support_ratio: seberapa sering pola muncul di seluruh sequence
    - confidence sekuensial: P(last | prefix)
    - lift sekuensial: confidence / support(last)
    """
    if df_pat is None or df_pat.empty or not sequences:
        return [], None

    total_seq = max(len(sequences), 1)

    # support item dasar (untuk lift): proporsi sequence yang mengandung item tsb
    unique_items = sorted({item for seq in sequences for item in seq})
    item_support = {it: sum(1 for seq in sequences if it in seq) / total_seq for it in unique_items}

    work = df_pat.copy()
    if "pattern_str" in work.columns:
        work["_pat_list"] = work["pattern_str"].apply(_pattern_to_list)
    elif "pattern" in work.columns:
        work["_pat_list"] = work["pattern"].apply(_pattern_to_list)
    else:
        work["_pat_list"] = work.iloc[:, 0].apply(_pattern_to_list)

    # Top-K berdasarkan support_count (kalau ada)
    if "support_count" in work.columns:
        top = work.sort_values("support_count", ascending=False).head(top_k)
    else:
        top = work.head(top_k)

    insights = []
    for _, row in top.iterrows():
        pat = row["_pat_list"]
        if len(pat) < 2:
            continue

        prefix = pat[:-1]
        last = pat[-1]

        supp_cnt = int(row["support_count"]) if "support_count" in row else _support_count(pat, sequences)
        supp_ratio = supp_cnt / total_seq

        prefix_cnt = _support_count(prefix, sequences)
        conf = (supp_cnt / prefix_cnt) if prefix_cnt > 0 else 0.0

        base = item_support.get(last, 0.0)
        lift = (conf / base) if base > 0 else float("nan")

        # label kekuatan hubungan (biar gampang dipahami)
        if math.isnan(lift):
            strength = "cukup kuat"
        elif lift >= 1.5:
            strength = "sangat kuat"
        elif lift >= 1.2:
            strength = "kuat"
        elif lift >= 1.0:
            strength = "cukup"
        else:
            strength = "lemah"

        prefix_str = " → ".join(prefix)
        pat_str = " → ".join(pat)

        insights.append(
            f"Jika pelanggan memiliki urutan pembelian **{prefix_str}**, maka sekitar **{conf*100:.2f}%** "
            f"cenderung diikuti oleh **{last}** (support pola **{supp_ratio*100:.2f}%**, total **{supp_cnt}** sequence). "
            f"Lift sekuensial **{lift:.2f}** → hubungan **{strength}**. "
            f"Implikasi: pola **{pat_str}** bisa dipakai untuk *rekomendasi/cross-sell berbasis urutan*."
        )

    top_view = top.copy()
    top_view["pattern_str"] = top_view["_pat_list"].apply(lambda x: " → ".join(x))
    cols_show = [c for c in ["pattern_str", "support_count", "support_ratio", "length"] if c in top_view.columns]
    top_view = top_view[cols_show] if cols_show else top_view[["pattern_str"]]

    return insights, top_view

def build_iforest_interpretation(hasil_if, top_k=3):
    """
    Interpretasi Top-K customer impulsif:
    tampilkan skor + 2 indikator numerik paling menonjol dibanding median customer Normal.
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
                b = baseline.get(c, float("nan"))
                v = r.get(c, float("nan"))
                if pd.isna(b) or pd.isna(v):
                    continue
                ratio = (float("inf") if (b == 0 and v != 0) else (1.0 if (b == 0 and v == 0) else v / b))
                ratios.append((c, ratio))
            ratios = sorted(ratios, key=lambda x: x[1], reverse=True)[:2]
            if ratios:
                highlight_txt = " Indikator menonjol: " + ", ".join([f"**{c}** ~{ratio:.2f}x median normal" for c, ratio in ratios]) + "."

        lines.append(f"Customer **{cid}** terdeteksi impulsif dengan skor anomali **{score:.4f}**.{highlight_txt}")

    return lines
# ===== End tambahan interpretasi =====

st.set_page_config(page_title="Dashboard PrefixSpan + Isolation Forest", layout="wide")

st.title("Dashboard Analisis Pola Pembelian & Deteksi Pelanggan Impulsif")
st.caption("Upload data transaksi → filter tahun & umur → PrefixSpan (pola sekuensial) "
            "& Isolation Forest (deteksi pelanggan impulsif & skor anomali).")

with st.sidebar:
    st.header("Input & Parameter")

uploaded = st.sidebar.file_uploader("Upload CSV transaksi", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV transaksi untuk memulai.")
    st.stop()

# Upload file CSV
try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

# Validasi kolom minimum sejak awal
if raw_df.shape[0] == 0:
    st.warning("File CSV kosong.")
    st.stop()

# --- Sidebar filter: rentang tahun dan rentang umur ---
with st.sidebar:
    st.subheader("Filter Tahun Transaksi")

# Deteksi tahun yang tersedia dari order_date
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

# Jika input tahun terbalik, otomatis dibetulkan
if year_start > year_end:
    year_start, year_end = year_end, year_start
    st.sidebar.info("Rentang tahun dibalik otomatis agar valid.")

with st.sidebar:
    st.subheader("Filter Umur Pelanggan")

# Deteksi rentang umur
age_col_guess = None
for cand in ["customer_age", "age", "umur", "usia"]:
    if cand in raw_df.columns:
        age_col_guess = cand
        break

# Jika tidak ada kolom umur, filter umur dimatikan
if age_col_guess is None:
    st.sidebar.warning("Kolom umur pelanggan tidak ditemukan. Filter umur dimatikan.")
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

     # Jika input umur terbalik, otomatis dibetulkan
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

# Jalankan pipeline
with st.spinner("Memproses data..."):
    df = clean_and_prepare(raw_df)

    df_filtered = df.copy()
    # Terapkan filter tahun (berdasarkan datetime yang sudah diparsing)
    df_filtered = df_filtered[df_filtered["order_year"].between(int(year_start), int(year_end))].copy()

     # Terapkan filter umur jika tersedia
    if age_min is not None and age_max is not None and "customer_age" in df_filtered.columns:
        df_filtered = df_filtered[pd.to_numeric(df_filtered["customer_age"], errors="coerce").between(int(age_min), int(age_max))].copy()

    # Hentikan jika data kosong setelah filter
    if df_filtered.empty:
        st.error("Data kosong setelah filter tahun/umur. Coba longgarkan range filternya.")
        st.stop()

    # Bentuk sekuens untuk PrefixSpan
    seq_df = build_sequences(df_filtered)
    sequences_all = seq_df["sequence"].tolist()

    # Bangun tabel order dan fitur customer untuk Isolation Forest
    order_tbl = build_order_table(df_filtered)
    cust_feat = build_customer_features(order_tbl)

    # Jalankan Isolation Forest untuk deteksi pelanggan impulsif
    hasil_if, if_info = run_isolation_forest(cust_feat, contamination=float(contamination), n_estimators=int(n_estimators),
                                             score_percentile=int(score_percentile), random_state=42,)

    # Jalankan PrefixSpan untuk menemukan pola sekuensial
    info_ps, df_pat = run_prefixspan(sequences_all, float(min_support_ratio), min_len=int(min_pattern_len))

# Tab tampilan
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "PrefixSpan", "Impulsif (IF)", "Normal vs Impulsif"])

with tab1:
    st.subheader("Ringkasan Data")
    cA, cB, cC = st.columns(3)
    cA.metric("Customer unik", int(df_filtered["customer_id"].nunique()))
    cB.metric("Order unik", int(df_filtered["order_id"].nunique()))
    cC.metric("Kategori unik", int(df_filtered["category"].nunique()))

    st.write("**Periode transaksi (setelah filter):**", f"{int(df_filtered['order_year'].min())} – {int(df_filtered['order_year'].max())}")
    if "customer_age" in df_filtered.columns:
        st.write("**Rentang umur (setelah filter):**", 
                 f"{int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').min())} –" 
                 f"{int(pd.to_numeric(df_filtered['customer_age'], errors='coerce').max())}")

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
    st.subheader("Interpretasi (Top 3)")
    insights, top_view = build_prefixspan_interpretation(df_pat, sequences_all, top_k=3)

    if not insights:
        st.info("Belum ada pola yang cukup untuk diinterpretasikan. Coba turunkan minimum support atau ubah filter.")
    else:
        for t in insights:
            st.markdown(f"- {t}")
        st.caption("Ringkasan Top 3 pola")
        st.dataframe(top_view, use_container_width=True)

        # opsional: unduh interpretasi
        st.download_button(
            "Download interpretasi (TXT)",
            data=("\n".join([x.replace('**','') for x in insights])).encode("utf-8"),
            file_name="interpretasi_prefixspan.txt",
            mime="text/plain",
        )

    st.download_button("Download pola", data=df_pat.to_csv(index=False).encode("utf-8"),
                       file_name="prefixspan_patterns.csv", mime="text/csv",)

with tab3:
    st.subheader("Hasil Isolation Forest (Deteksi Pelanggan Impulsif)")
    st.dataframe(if_info, use_container_width=True)
    st.subheader("Interpretasi (Top 3 pelanggan impulsif)")
    for t in build_iforest_interpretation(hasil_if, top_k=3):
        st.markdown(f"- {t}")

    st.caption("Daftar customer (urut skor anomali tertinggi)")
    st.dataframe(hasil_if.sort_values("anom_score", ascending=False), use_container_width=True, height=520)

    st.download_button("Download pelanggan impulsif", data=hasil_if.to_csv(index=False).encode("utf-8"),
                       file_name="impulsive_customers_iforest.csv", mime="text/csv",)

with tab4:
    st.subheader("Perbandingan Pola Normal vs Impulsif")

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
            st.download_button("Download pola normal", data=pat_n.to_csv(index=False).encode("utf-8"),
                               file_name="prefixspan_patterns_normal.csv", mime="text/csv",)

    with colR:
        st.markdown("### Impulsif")
        if len(imp_seqs) < MIN_CUSTOMERS_PATTERN:
            st.info(f"Sequence impulsif terlalu sedikit ({len(imp_seqs)}). Minimal {MIN_CUSTOMERS_PATTERN} untuk mining pola.")
        else:
            info_i, pat_i = run_prefixspan(imp_seqs, float(min_support_ratio), min_len=int(min_pattern_len))
            st.dataframe(info_i, use_container_width=True)
            st.dataframe(pat_i, use_container_width=True, height=420)
            st.download_button("Download pola impulsif", data=pat_i.to_csv(index=False).encode("utf-8"),
                               file_name="prefixspan_patterns_impulsive.csv", mime="text/csv",)
