# Streamlit Deployment — PrefixSpan + Isolation Forest

Ini project siap jalan di VS Code (lokal) buat deployment hasil riset kamu.

## 1) Struktur folder

- `app.py` → Streamlit UI
- `src/preprocess.py` → cleaning + (opsional) segmentasi Gen Z + pembentukan sequence + fitur customer-level
- `src/mining.py` → PrefixSpan (pola sekuensial)
- `src/anomaly.py` → Isolation Forest (deteksi impulsif)
- `requirements.txt` → dependency

## 2) Format CSV yang diharapkan

Minimal kolom ini (case-sensitive):

- `customer_id`
- `order_id`
- `order_date` (format bebas, akan diparse `pd.to_datetime`)
- `category`
- `total_amount`
- `quantity`
- `discount`
- `customer_age`

Kolom lain boleh ada — akan diabaikan.


> Bonus: app juga coba **auto-rename** beberapa nama kolom umum (mis. `order_total` → `total_amount`). Kalau belum kena, tinggal rename manual.

## 3) Cara run di VS Code (lokal)

```bash
# (opsional) bikin virtual env
python -m venv .venv

# activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Nanti otomatis kebuka di browser.

## 4) Output aplikasi

- **PrefixSpan**: tabel pola sekuensial + support
- **Impulsif (IF)**: list customer impulsif + `anom_score` (makin besar = makin anomali)
- **Normal vs Impulsif**: perbandingan pola sekuensial antara dua kelompok

## 5) Catatan penting

- Default aplikasi = **semua usia**.
- Kamu bisa nyalain **segmentasi Gen Z** lewat checkbox di sidebar.
- Definisi Gen Z dihitung dari perkiraan tahun lahir: `order_year - customer_age` berada di **1997–2012**.
  Ini ekuivalen dengan rentang umur per tahun di notebook (2023/2024/2025).
- Kalau pola impulsif terlalu sedikit → turunin `minimum support` atau naikin `max panjang sequence`.


## Filter Tahun Transaksi
- Default: semua tahun transaksi (tidak dibatasi).
- Kamu bisa nonaktifkan opsi "Gunakan semua tahun transaksi" lalu pilih tahun tertentu di sidebar.
