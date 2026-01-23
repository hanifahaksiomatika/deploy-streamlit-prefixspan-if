import pandas as pd
import numpy as np

# ---------- Column aliasing ----------
ALIASES = {
    "customer_id": ["customer_id", "cust_id", "customer", "id_customer", "id_pelanggan"],
    "order_id": ["order_id", "transaction_id", "trx_id", "invoice_id", "id_order", "id_transaksi"],
    "order_date": ["order_date", "tanggal", "tanggal_transaksi", "date", "trx_date"],
    "category": ["category", "kategori", "product_category", "kategori_produk"],
    "total_amount": ["total_amount", "amount", "total", "order_total", "order_amount", "order_value", "harga_total"],
    "quantity": ["quantity", "qty", "jumlah", "items", "item_count"],
    "discount": ["discount", "diskon", "discount_amount", "potongan"],
    "customer_age": ["customer_age", "age", "umur", "usia"],
}

def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)
    mapping = {}
    for target, candidates in ALIASES.items():
        for c in candidates:
            if c in cols:
                mapping[c] = target
                break
    return df.rename(columns=mapping)

def clean_and_prepare(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning minimal mengikuti notebook:
    - rename kolom (alias)
    - parse order_date
    - normalisasi category
    - filter total_amount > 0 dan quantity > 0
    - buat has_discount dan order_year
    """
    df = _rename_with_aliases(raw_df)

    required = ["customer_id", "order_id", "order_date", "category", "total_amount", "quantity", "discount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}. "
                         f"Pastikan CSV punya kolom minimal: {required} (customer_age opsional).")

    # parse date
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]).copy()

    # normalize category
    df["category"] = df["category"].astype(str).str.strip()

    # numeric
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0)

    df = df[(df["total_amount"] > 0) & (df["quantity"] > 0)].copy()

    # age optional
    if "customer_age" in df.columns:
        df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")

    df["has_discount"] = (df["discount"].fillna(0) > 0).astype(int)
    df["order_year"] = df["order_date"].dt.year.astype(int)

    # Sort for reproducibility
    df = df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)

    return df

def collapse_consecutive(items):
    out = []
    for v in items:
        if len(out) == 0 or v != out[-1]:
            out.append(v)
    return out

def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequence per customer: list category ordered by time + collapse consecutive duplicates.
    """
    df_seq = df.sort_values(["customer_id", "order_date", "order_id"]).copy()
    seq_df = df_seq.groupby("customer_id")["category"].apply(list).reset_index(name="sequence")
    seq_df["sequence"] = seq_df["sequence"].apply(collapse_consecutive)
    seq_df["sequence_length"] = seq_df["sequence"].apply(len)
    return seq_df

def build_order_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order-level table mengikuti notebook:
    groupby customer_id, order_id:
      - order_date min
      - order_total sum(total_amount)
      - items sum(quantity)
      - has_discount max
    """
    order_tbl = (
        df.groupby(["customer_id", "order_id"])
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

def build_customer_features(order_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Customer-level features mengikuti notebook:
    jml_order, total_spend, avg_order, std_order, avg_items, discount_rate, avg_gap_days
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
    cust_feat["avg_gap_days"] = cust_feat["avg_gap_days"].fillna(cust_feat["avg_gap_days"].median() if cust_feat["avg_gap_days"].notna().any() else 0)

    return cust_feat
