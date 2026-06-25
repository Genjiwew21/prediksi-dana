# 📱 Dashboard Analisis Sentimen DANA (Streamlit)

Dashboard interaktif untuk menampilkan hasil analisis sentimen ulasan
pengguna aplikasi **DANA** di Google Play Store, menggunakan algoritma
**Naïve Bayes**, **SVM**, dan **Random Forest** — diadaptasi dari notebook
`Analisis_Sentimen_DANA_Final_Clean.ipynb`.

---

## 📁 Struktur Folder

```
dana_sentiment_app/
├── app.py                 # Aplikasi utama Streamlit (dashboard)
├── train_model.py         # Script training, menghasilkan folder models/
├── data_dana.csv          # Dataset mentah (15.089+ baris ulasan)
├── requirements.txt       # Daftar library Python yang dibutuhkan
├── packages.txt           # Dependency sistem (untuk Streamlit Cloud)
├── .gitignore
├── README.md               # File ini
└── models/                 # (dihasilkan otomatis setelah training)
    ├── tfidf_vectorizer.pkl
    ├── model_naive_bayes.pkl
    ├── model_svm.pkl
    ├── model_random_forest.pkl
    ├── metrics.pkl
    ├── confusion_matrices.pkl
    └── sample_data.pkl
```

---

## 🚀 Cara Menjalankan di Komputer Lokal

### 1. Buat virtual environment (sangat disarankan)

```bash
python -m venv venv

# Aktifkan:
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install semua library yang dibutuhkan

```bash
pip install -r requirements.txt
```

> Jika muncul error terkait `PySastrawi`, pastikan koneksi internet aktif
> saat instalasi, karena package ini diambil dari PyPI.

### 3. Latih model (HARUS dijalankan sekali sebelum app dibuka)

```bash
python train_model.py
```

Proses ini akan:
- Membaca `data_dana.csv`
- Melakukan labeling otomatis berdasarkan Rating
- Melakukan preprocessing teks (cleaning, case folding, tokenizing,
  stopword removal, stemming menggunakan Sastrawi)
- Melatih 3 model: Naïve Bayes, SVM, Random Forest
- Menyimpan semua model + metrik ke folder `models/`

⏱️ **Estimasi waktu:** untuk ±15.000 baris data, proses stemming bisa
memakan waktu **5–15 menit** tergantung spesifikasi komputer (stemming
Bahasa Indonesia memang relatif berat secara komputasi). Setelah selesai
sekali, kamu tidak perlu mengulanginya lagi kecuali datanya berubah.

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Browser otomatis terbuka di `http://localhost:8501`.

---

## ☁️ Cara Deploy ke Streamlit Community Cloud (Gratis)

1. **Buat repository GitHub baru**, lalu upload seluruh isi folder ini
   (termasuk `data_dana.csv`, dan **folder `models/` hasil training**
   supaya tidak perlu training ulang di server — ini sangat disarankan
   karena training di cloud bisa timeout/lambat).

   > Jika ukuran file `data_dana.csv` atau model terlalu besar untuk
   > GitHub (limit 100MB per file, atau repo gratis 1GB), gunakan
   > [Git LFS](https://git-lfs.com/) atau cukup upload folder `models/`
   > saja tanpa CSV mentah (karena app.py hanya butuh folder `models/`
   > untuk berjalan normal — CSV hanya dipakai saat training).

2. Buka **[share.streamlit.io](https://share.streamlit.io)**, login
   dengan akun GitHub.

3. Klik **"New app"**, pilih:
   - Repository: repo yang baru dibuat
   - Branch: `main`
   - Main file path: `app.py`

4. Klik **Deploy**. Streamlit Cloud akan otomatis:
   - Membaca `requirements.txt` → install semua library Python
   - Membaca `packages.txt` → install dependency sistem (jika ada)
   - Menjalankan `app.py`

5. Tunggu beberapa menit sampai status menjadi **"Your app is live!"** 🎉

### ⚠️ Catatan penting untuk deploy ke cloud

- **Selalu sertakan folder `models/`** hasil training di repo. Jangan
  hanya mengandalkan `train_model.py` dijalankan otomatis di server
  cloud, karena:
  - Server gratis Streamlit Cloud punya limit memori/waktu
  - Proses stemming Sastrawi untuk ribuan baris bisa membuat aplikasi
    timeout sebelum sempat tampil
- Jika repo terlalu besar, kamu juga bisa upload folder `models/` ke
  Google Drive lalu memodifikasi `app.py` agar mengunduhnya otomatis
  saat pertama kali dijalankan (opsional, hubungi pembuat kode jika
  butuh bantuan versi ini).

---

## 🖥️ Fitur Dashboard

| Halaman | Isi |
|---|---|
| 🏠 Ringkasan | Statistik dataset, distribusi label, ringkasan performa 3 model |
| 📊 Eksplorasi Data | Tabel data hasil preprocessing, word cloud per sentimen |
| 🤖 Perbandingan Model | Tabel & grafik metrik (Accuracy/Precision/Recall/F1), confusion matrix, classification report |
| 🔮 Coba Prediksi Sendiri | Input ulasan bebas → prediksi sentimen real-time dengan model pilihan |

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---|---|
| Error `Preparing metadata (pyproject.toml) ... error` + `Could not find vswhere.exe` saat install pandas | Kamu memakai **Python 3.14** dan versi pandas di `requirements.txt` masih versi lama yang belum punya installer siap pakai (wheel) untuk 3.14, jadi pip mencoba compile dari source dan butuh Visual Studio Build Tools. **Solusi termudah:** `requirements.txt` di paket ini sudah diperbarui untuk pakai versi minimum (`>=`) bukan versi terkunci (`==`), jadi cukup jalankan ulang `pip install -r requirements.txt`. Jika masih gagal, install **Python 3.12** dari [python.org](https://www.python.org/downloads/release/python-3120/), buat ulang venv dengan `py -3.12 -m venv venv`, lalu ulangi langkah instalasi — Python 3.12 didukung penuh oleh semua library data science. |
| `ModuleNotFoundError: No module named 'numpy'` saat `python train_model.py` | Berarti `pip install -r requirements.txt` di langkah sebelumnya **gagal di tengah jalan** (biasanya karena error pandas di atas). Pastikan instalasi `requirements.txt` selesai 100% tanpa error sebelum lanjut ke `train_model.py`. |
| `streamlit: The term 'streamlit' is not recognized` | Artinya streamlit belum terinstal (konsekuensi dari kegagalan `pip install` di atas), **atau** venv belum aktif. Pastikan prompt terminal menunjukkan `(venv)` di depan sebelum menjalankan `streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'Sastrawi'` | Jalankan `pip install PySastrawi` |
| Model belum ada / app menampilkan pesan error model | Jalankan `python train_model.py` terlebih dahulu |
| Training sangat lambat | Normal untuk dataset besar; bisa kurangi jumlah data di `train_model.py` untuk uji coba cepat (`df = df.sample(3000)` setelah load data) |
| Error saat baca CSV (`UnicodeDecodeError`) | Pastikan file `data_dana.csv` tidak diedit ulang formatnya; gunakan file asli hasil scraping |
| Word cloud tidak muncul | Pastikan `wordcloud` ada di `requirements.txt` dan sudah terinstall |
| Streamlit Cloud: aplikasi "sleeping"/lambat pertama dibuka | Wajar untuk free tier; tunggu beberapa detik untuk wake up |

---

## 📚 Referensi Algoritma

- **Naïve Bayes** (ComplementNB) — cocok untuk data teks dengan distribusi
  kelas yang tidak seimbang.
- **SVM** (LinearSVC) — umumnya kuat untuk klasifikasi teks berdimensi
  tinggi seperti TF-IDF.
- **Random Forest** — ensemble tree-based, baik dalam menangani fitur
  yang kompleks namun lebih berat secara komputasi.

---

Dibuat untuk mendukung Tugas Akhir / Skripsi:
**Analisis Sentimen Ulasan Pengguna Aplikasi DANA Menggunakan Algoritma
Naïve Bayes, SVM, dan Random Forest** — Universitas Multimedia Nusantara.
