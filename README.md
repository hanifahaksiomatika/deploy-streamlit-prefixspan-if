Link Streamlit: https://deploy-app-prefixspan-if.streamlit.app/

# Dashboard PrefixSpan + Isolation Forest (Streamlit)

Aplikasi ini mengimplementasikan deployment penelitian:
- **PrefixSpan** untuk mengekstraksi pola sekuensial pembelian (sequence per customer, token kategori + status diskon).
- **Isolation Forest** untuk mendeteksi pelanggan impulsif berbasis fitur agregat per pelanggan dan skor anomali.

## Update ke repo GitHub lama
Copy/overwrite file `app.py`, folder `src/`, dan `requirements.txt` ke folder repo lama, lalu:
- `git add .`
- `git commit -m "Update dashboard"`
- `git pull origin main`
- `git push origin main`

Streamlit Cloud akan redeploy otomatis jika repo sudah terhubung.
