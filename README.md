# Streamlit Deployment — PrefixSpan + Isolation Forest (Range Filter)

Aplikasi Streamlit untuk:
- Mining pola sekuensial transaksi (PrefixSpan)
- Deteksi pelanggan impulsif + skor anomali (Isolation Forest)

Fitur filter:
- Range tahun transaksi (Dari–Sampai)
- Range umur pelanggan (Min–Max)

## Menjalankan di Lokal (Anaconda)
```bash
cd /d C:\Hani\skripsi\deploy_streamlit_riset_utama_baru_rangefilters
conda create -n skripsi_streamlit python=3.10 -y
conda activate skripsi_streamlit
pip install -r requirements.txt
streamlit run app.py
```

Buka: http://localhost:8501

## Kolom CSV Minimal
- customer_id
- order_id
- order_date
- category
- total_amount
- quantity
- discount
- customer_age (opsional, tapi dibutuhkan untuk filter umur)

Catatan: beberapa nama kolom umum akan di-*auto-map* (alias) di `src/preprocess.py`.