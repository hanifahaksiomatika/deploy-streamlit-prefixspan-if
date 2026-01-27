# Dashboard PrefixSpan + Isolation Forest (Streamlit)

Aplikasi ini mengimplementasikan deployment penelitian:
- **PrefixSpan** untuk mengekstraksi pola sekuensial pembelian (sequence per customer, token kategori + status diskon).
- **Isolation Forest** untuk mendeteksi pelanggan impulsif berbasis fitur agregat per pelanggan dan skor anomali.

## Menjalankan via Anaconda (Windows)

1. Buka **Anaconda Prompt**
2. Masuk folder project:
   - `cd /d C:\Hani\skripsi\deploy_streamlit_prefixspan_if`
3. Buat & aktifkan environment:
   - `conda create -n skripsi_streamlit python=3.10 -y`
   - `conda activate skripsi_streamlit`
4. Install dependency:
   - `pip install -r requirements.txt`
5. Jalankan:
   - `streamlit run app.py`

## Update ke repo GitHub lama
Copy/overwrite file `app.py`, folder `src/`, dan `requirements.txt` ke folder repo lama, lalu:
- `git add .`
- `git commit -m "Update dashboard"`
- `git pull origin main`
- `git push origin main`

Streamlit Cloud akan redeploy otomatis jika repo sudah terhubung.
