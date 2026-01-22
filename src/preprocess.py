from __future__ import annotations

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "customer_id",
    "order_id",
    "order_date",
    "category",
    "total_amount",
    "quantity",
    "discount",
    "customer_age",
]



# Alias mapping (optional): kalau kolom dataset kamu beda dikit, app bakal coba auto-rename.
# Kamu bisa tambah sendiri sesuai kebutuhan.
COLUMN_ALIASES = {
    "customer_id": ["customerid", "cust_id", "customer", "id_customer"],
    "order_id": ["orderid", "invoice_id", "transaction_id", "trx_id"],
    "order_date": ["orderdate", "date", "transaction_date", "trx_date"],
    "category": ["product_category", "kategori", "category_name"],
    "total_amount": ["order_total", "total", "amount", "total_price", "grand_total"],
    "quantity": ["qty", "jumlah", "item_qty"],
    "discount": ["disc", "discount_amount", "promo"],
    "customer_age": ["age", "umur"],
}

def normalize_columns(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Return a copy of df with columns normalized (lowercase + strip) and auto-renamed using COLUMN_ALIASES.
    It only renames when there is an unambiguous match.
    """
    df = df_raw.copy()
    # map lowercase->original
    lc_map = {c.lower().strip(): c for c in df.columns}
    renames: dict[str, str] = {}

    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        # find alias present
        hits = [a for a in aliases if a in lc_map]
        if len(hits) == 1:
            src_col = lc_map[hits[0]]
            renames[src_col] = target

    if renames:
        df = df.rename(columns=renames)

    return df, renames
def validate_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    return missing

def _to_numeric(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")

def clean_transactions(
    df_raw: pd.DataFrame,
    *,
    filter_genz: bool = False,
    keep_years: list[int] | None = None,
    genz_birth_year_min: int = 1997,
    genz_birth_year_max: int = 2012,
) -> pd.DataFrame:
    """Cleaning pipeline yang *selaras* sama notebook, tapi lebih fleksibel.

    Steps:
    - Parse order_date
    - Normalize category
    - Filter total_amount > 0 & quantity > 0
    - Feature has_discount (discount > 0)
    - Optional: filter tahun tertentu (keep_years)
    - Optional: segmentasi Gen Z

    Catatan Gen Z:
    Notebook kamu pakai rentang umur per tahun (2023/2024/2025). Cara ini ekuivalen
    dengan filter *perkiraan tahun lahir* (order_year - age) berada di 1997–2012.
    """
    df = df_raw.copy()

    # Type fixes
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()

    _to_numeric(df, "total_amount")
    _to_numeric(df, "quantity")
    _to_numeric(df, "discount")
    _to_numeric(df, "customer_age")

    # Normalize category
    df["category"] = df["category"].astype(str).str.strip()

    # Remove weird values
    df = df[(df["total_amount"] > 0) & (df["quantity"] > 0)].copy()

    # Discount feature
    df["has_discount"] = (df["discount"].fillna(0) > 0).astype(int)

    # Year
    df["order_year"] = df["order_date"].dt.year
    if keep_years is not None:
        df = df[df["order_year"].isin(keep_years)].copy()

    # Optional Gen Z segmentation
    if filter_genz:
        birth_year = df["order_year"] - df["customer_age"]
        genz_mask = birth_year.between(genz_birth_year_min, genz_birth_year_max)
        df = df[genz_mask].copy()

    # Sort for stable sequences
    df = df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    return df


# Backward-compat wrapper (biar import lama nggak rusak)
def clean_and_filter_genz(df_raw: pd.DataFrame) -> pd.DataFrame:
    return clean_transactions(df_raw, filter_genz=True, keep_years=[2023, 2024, 2025])

def collapse_consecutive(x: list) -> list:
    out = []
    for v in x:
        if not out or v != out[-1]:
            out.append(v)
    return out

def build_sequences_df(df_genz: pd.DataFrame, min_seq: int = 2, max_seq: int = 5) -> pd.DataFrame:
    """
    Build a sequence of categories per customer, collapse consecutive duplicates,
    and filter by sequence length.
    """
    df_seq = df_genz.sort_values(["customer_id", "order_date", "order_id"]).copy()
    seq_df = (
        df_seq.groupby("customer_id")["category"]
        .apply(list)
        .reset_index(name="sequence")
    )
    seq_df["sequence"] = seq_df["sequence"].apply(collapse_consecutive)
    seq_df["sequence_length"] = seq_df["sequence"].apply(len)
    seq_df = seq_df[(seq_df["sequence_length"] >= min_seq) & (seq_df["sequence_length"] <= max_seq)].copy()
    return seq_df

def build_order_table(df_genz: pd.DataFrame) -> pd.DataFrame:
    """
    Order-level aggregation (like notebook):
    group by customer_id + order_id
    """
    order_tbl = (
        df_genz.groupby(["customer_id", "order_id"])
        .agg(
            order_date=("order_date", "min"),
            order_total=("total_amount", "sum"),
            items=("quantity", "sum"),
            has_discount=("has_discount", "max"),
        )
        .reset_index()
        .sort_values(["customer_id", "order_date", "order_id"])
        .reset_index(drop=True)
    )
    return order_tbl

def build_customer_features(order_tbl: pd.DataFrame):
    """
    Customer-level features (like notebook):
    - jml_order, total_spend, avg_order, std_order, avg_items,
      discount_rate, avg_gap_days
    """
    tmp = order_tbl.sort_values(["customer_id", "order_date"]).copy()
    tmp["gap_days"] = tmp.groupby("customer_id")["order_date"].diff().dt.days

    cust_feat = (
        tmp.groupby("customer_id")
        .agg(
            jml_order=("order_id", "nunique"),
            total_spend=("order_total", "sum"),
            avg_order=("order_total", "mean"),
            std_order=("order_total", "std"),
            avg_items=("items", "mean"),
            discount_rate=("has_discount", "mean"),
            avg_gap_days=("gap_days", "mean"),
        )
        .reset_index()
    )

    cust_feat["std_order"] = cust_feat["std_order"].fillna(0)
    cust_feat["avg_gap_days"] = cust_feat["avg_gap_days"].fillna(0)

    feature_cols = ["jml_order","total_spend","avg_order","std_order","avg_items","discount_rate","avg_gap_days"]
    return cust_feat, feature_cols
