# Dashboard Analisis Pola Pembelian & Deteksi Pelanggan Impulsif  
**PrefixSpan (Sequential Pattern Mining) + Isolation Forest (Anomaly Detection)**

🚀 **Live App (Streamlit):** https://deploy-app-prefixspan-if.streamlit.app/

Aplikasi ini merupakan implementasi *deployment* dari penelitian skripsi untuk:
1) mengekstraksi **pola sekuensial pembelian** menggunakan **PrefixSpan**, dan  
2) mendeteksi **pelanggan impulsif** menggunakan **Isolation Forest** dengan **skor anomali**.

---

## Fitur Utama
- **Upload CSV transaksi** (tanpa perlu database)
- **(Opsional) Segmentasi Gen Z** melalui filter usia
- **(Opsional) Filter tahun transaksi** (atau gunakan semua tahun)
- Output:
  - **Pola sekuensial (PrefixSpan)**: urutan kategori yang sering muncul beserta support
  - **Deteksi pelanggan impulsif (Isolation Forest)**: daftar customer dengan label anomali dan skor anomalinya
- **Download hasil** ke CSV untuk lampiran/analisis lanjutan

---

## Alur Sistem
1. Input data transaksi (CSV)  
2. Pembersihan data + validasi kolom  
3. (Opsional) Filter Gen Z  
4. Pembentukan *sequence* per pelanggan berdasarkan urutan transaksi  
5. Ekstraksi pola sekuensial dengan PrefixSpan  
6. Agregasi fitur perilaku pembelian per pelanggan  
7. Deteksi impulsif menggunakan Isolation Forest → skor anomali  
8. Output ditampilkan pada dashboard + file hasil dapat diunduh

---

## Format Data (CSV)
Minimal kolom yang dibutuhkan:

| Kolom | Deskripsi |
|------|-----------|
| `customer_id` | ID pelanggan |
| `order_id` | ID transaksi/order |
| `order_date` | tanggal transaksi (disarankan format `YYYY-MM-DD`) |
| `category` | kategori produk |
| `total_amount` | total belanja per transaksi |
| `quantity` | jumlah item |
| `discount` | nilai diskon (0 jika tidak ada) |
| `customer_age` | usia pelanggan (angka) |

> **Catatan:** jika nama kolom di dataset kamu berbeda, silakan samakan/rename ke format di atas agar aplikasi bisa memproses dengan benar.
