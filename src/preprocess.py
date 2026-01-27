import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "customer_id",
    "order_id",
    "order_date",
    "category",
    "total_amount",
    "quantity",
    "discount",
]

def clean_and_prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning supaya konsisten dengan notebook riset_utama.ipynb:
    - pastikan kolom ada (rename beberapa alias umum)
    - parse order_date → order_year
    - has_discount = discount > 0
    - normalisasi tipe data dasar
    """
    df = raw_df.copy()

    # alias mapping (biar fleksibel)
    rename_map = {
        "cust_id": "customer_id",
        "customer": "customer_id",
        "id_customer": "customer_id",
        "order": "order_id",
        "orderid": "order_id",
        "tanggal": "order_date",
        "date": "order_date",
        "product_category": "category",
        "kategori": "category",
        "amount": "total_amount",
        "total": "total_amount",
        "qty": "quantity",
        "jumlah": "quantity",
        "disc": "discount",
        "potongan": "discount",
        "umur": "customer_age",
        "usia": "customer_age",
        "age": "customer_age",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # validate minimal columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. Pastikan CSV punya minimal kolom {REQUIRED_COLS}.")

    # parse date
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()
    df["order_year"] = df["order_date"].dt.year

    # numeric coercion
    for c in ["total_amount", "quantity", "discount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # age optional
    if "customer_age" in df.columns:
        df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")

    # has_discount
    df["has_discount"] = (df["discount"] > 0).astype(int)

    # tidy strings
    df["category"] = df["category"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["order_id"] = df["order_id"].astype(str).str.strip()

    return df

def apply_filters(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    age_min: int | None,
    age_max: int | None,
    segment: str,
) -> pd.DataFrame:
    out = df.copy()

    # tahun transaksi
    if "order_year" in out.columns:
        out = out[out["order_year"].between(int(year_start), int(year_end))].copy()

    # segmen Gen Z berdasarkan aturan tahun transaksi (sesuai notebook)
    if segment.startswith("Gen Z"):
        if "customer_age" in out.columns and "order_year" in out.columns:
            genz = (
                ((out["order_year"] == 2023) & (out["customer_age"].between(11, 26))) |
                ((out["order_year"] == 2024) & (out["customer_age"].between(12, 27))) |
                ((out["order_year"] == 2025) & (out["customer_age"].between(13, 28)))
            )
            out = out[genz].copy()
        else:
            # kalau nggak ada umur, kita nggak bisa apply Gen Z
            out = out.iloc[0:0].copy()

    # umur custom (kalau ada)
    if age_min is not None and age_max is not None and "customer_age" in out.columns:
        age_num = pd.to_numeric(out["customer_age"], errors="coerce")
        out = out[age_num.between(int(age_min), int(age_max))].copy()

    return out

def build_order_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order-level table sesuai notebook:
    groupby customer_id, order_id:
      - order_date min
      - order_total sum(total_amount)
      - items sum(quantity)
      - has_discount max
      - avg_discount mean(discount)
    + gap_days per customer
    """
    order_tbl = (
        df.groupby(["customer_id", "order_id"])
        .agg(
            order_date=("order_date", "min"),
            order_total=("total_amount", "sum"),
            items=("quantity", "sum"),
            has_discount=("has_discount", "max"),
            avg_discount=("discount", "mean"),
        )
        .reset_index()
        .sort_values(["customer_id", "order_date", "order_id"])
        .reset_index(drop=True)
    )

    order_tbl["gap_days"] = order_tbl.groupby("customer_id")["order_date"].diff().dt.days
    return order_tbl

def build_customer_features(order_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur customer-level (mengikuti notebook):
      jml_order, total_spend, avg_order, std_order, avg_items, avg_gap_days,
      discount_order_ratio, avg_discount_rate
    """
    tmp = order_tbl.copy()
    if "gap_days" in tmp.columns:
        med_gap = tmp["gap_days"].median()
        tmp["gap_days"] = tmp["gap_days"].fillna(med_gap if pd.notna(med_gap) else 0)

    cust_feat = (
        tmp.groupby("customer_id")
        .agg(
            jml_order=("order_id", "nunique"),
            total_spend=("order_total", "sum"),
            avg_order=("order_total", "mean"),
            std_order=("order_total", "std"),
            avg_items=("items", "mean"),
            avg_gap_days=("gap_days", "mean"),
            discount_order_ratio=("has_discount", "mean"),
            avg_discount_rate=("avg_discount", "mean"),
        )
        .reset_index()
    )
    cust_feat["std_order"] = cust_feat["std_order"].fillna(0)
    return cust_feat

def build_sequences(df_filtered: pd.DataFrame, hasil_if: pd.DataFrame, min_seq_len: int = 2) -> pd.DataFrame:
    """
    Sequence per customer TANPA collapse, menggunakan token category_D / category_ND
    seperti notebook (cat_token).
    """
    df_seq = df_filtered.sort_values(["customer_id", "order_date", "order_id"]).copy()

    df_seq["has_discount"] = df_seq["has_discount"].astype(int)
    df_seq["cat_token"] = np.where(
        df_seq["has_discount"] == 1,
        df_seq["category"].astype(str).str.strip() + "_D",
        df_seq["category"].astype(str).str.strip() + "_ND",
    )

    # merge label IF (status & anom_score)
    if hasil_if is not None and not hasil_if.empty and "customer_id" in hasil_if.columns:
        df_seq = df_seq.merge(hasil_if[["customer_id", "status", "anom_score"]], on="customer_id", how="left")
    else:
        df_seq["status"] = "Unknown"
        df_seq["anom_score"] = np.nan

    seq_df = df_seq.groupby(["customer_id", "status"])["cat_token"].apply(list).reset_index(name="sequence")
    seq_df["sequence_length"] = seq_df["sequence"].apply(len)
    seq_df = seq_df[seq_df["sequence_length"] >= int(min_seq_len)].copy()
    return seq_df
