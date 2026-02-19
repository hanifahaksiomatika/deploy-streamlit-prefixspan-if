Link Streamlit: https://deploy-app-prefixspan-if.streamlit.app/

# Dashboard Ringkas (Skripsi) — PrefixSpan + Isolation Forest

- Filter **tahun** & **umur**
- Input **minimum panjang sequence customer**
- Output: **Interpretasi Impulsif**, **Pola Impulsif vs Normal**, **Diskon vs Non-Diskon**, dan **runtime**

## Run (Anaconda)
```bash
conda env create -f environment.yml
conda activate skripsi_streamlit
streamlit run app.py
```

## Run (pip)
```bash
pip install -r requirements.txt
streamlit run app.py
```
