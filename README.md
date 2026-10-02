# Analisis Sentimen Ulasan Roblox Menggunakan CNN (Convolutional Neural Network) & Klasifikasi Spesifikasi Aspek

Repositori ini berisi implementasi pipeline analisis sentimen untuk ulasan aplikasi Roblox di Google Play Store menggunakan arsitektur deep learning **Convolutional Neural Network (CNN)** 1D, deteksi sentimen ambigu, serta modul **kategorisasi aspek/spesifikasi ulasan** berbasis kamus kata kunci kontekstual.

---

## 📌 Daftar Pustaka & Dependensi yang Harus Diinstal

Proyek ini dibangun menggunakan Python (disarankan **Python 3.8 – 3.11**).

### 1. Instalasi Otomatis via `requirements.txt`
Jalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

### 2. Rincian Pustaka (Library) yang Digunakan
Jika ingin menginstal secara manual atau mengetahui fungsi masing-masing modul:

| Pustaka | Perintah Instal | Fungsi / Kegunaan |
| :--- | :--- | :--- |
| **TensorFlow / Keras** | `pip install tensorflow` | Pembangunan arsitektur model CNN, tokenizer kata, padding teks, dan proses pelatihan model (*training*). |
| **google-play-scraper** | `pip install google-play-scraper` | Mengambil (*scraping*) ulasan pengguna aplikasi Roblox langsung dari Google Play Store. |
| **googletrans** | `pip install googletrans==4.0.0rc1` | Menerjemahkan ulasan dari bahasa asing ke Bahasa Indonesia. |
| **PySastrawi** | `pip install PySastrawi` | Stemming kata berbahasa Indonesia untuk mengubah kata berimbuhan menjadi kata dasar. |
| **NLTK** | `pip install nltk` | Pemrosesan bahasa alami (NLP) seperti tokenisasi kalimat/kata (*punkt*). |
| **scikit-learn** | `pip install scikit-learn` | Pembagian dataset (*train-val-test split*), penghitungan *classification report*, dan *confusion matrix*. |
| **Pandas** | `pip install pandas` | Manipulasi dan penyimpanan data dalam format tabular (DataFrame / CSV). |
| **NumPy** | `pip install numpy` | Komputasi array numerik dan operasi probabilitas matriks. |
| **Matplotlib** | `pip install matplotlib` | Visualisasi grafik *learning curve*, distribusi label, dan grafik batang. |
| **Seaborn** | `pip install seaborn` | Pembuatan visualisasi visual heatmap *confusion matrix*. |
| **Tabulate** | `pip install tabulate` | Menampilkan tabel metrik evaluasi dan ringkasan dengan rapi di terminal console. |
| **tqdm** | `pip install tqdm` | Menampilkan *progress bar* selama proses iterasi data berlangsung. |

---

## 📁 Struktur Direktori & Penjelasan Folder

```text
analisis-roblox-cnn/
├── data/                                 # Folder dataset & semua file hasil proses / grafik
│   ├── reviews_raw.json
│   ├── reviews_translated.csv
│   ├── reviews_clean.csv
│   ├── split_data.csv
│   ├── classification_report.csv
│   ├── summary_metrics.csv
│   ├── confusion_matrix.csv
│   ├── evaluasi.csv
│   ├── prediksi_all.csv
│   ├── prediksi_all_spesifikasi.csv      # Output dari spesifikasi.py
│   ├── ambigu_summary.csv
│   ├── learning_curve.png
│   ├── confusion_matrix.png
│   ├── distribusi_label.png
│   └── distribusi_ambigu.png
├── models/                               # Folder bobot model deep learning
│   └── model.h5
├── .gitignore                            # Konfigurasi filter file Git
├── main.py                               # Script utama: scraping, CNN training, evaluasi & prediksi
├── spesifikasi.py                        # Script kategorisasi aspek/spesifikasi ulasan
├── requirements.txt                      # Daftar dependensi modul Python
└── README.md                             # Dokumentasi proyek
```

---

### 📂 Isi Folder `data/`
Folder `data/` digunakan untuk menyimpan seluruh berkas keluaran (*output*), mulai dari data mentah, data olahan, hasil prediksi, hingga visualisasi grafik.

> **Catatan Git:** Semua file hasil di dalam folder `data/` telah diatur di [.gitignore](.gitignore) agar tidak ter-push ke GitHub dan membebani repositori.

| Nama File | Deskripsi / Penjelasan |
| :--- | :--- |
| `reviews_raw.json` | Data ulasan mentah hasil *scraping* dari Google Play Store (format JSON). |
| `reviews_translated.csv` | Ulasan yang telah diterjemahkan ke Bahasa Indonesia beserta rating bintang dan label awal. |
| `reviews_clean.csv` | Dataset ulasan yang telah melewati tahap pembersihan teks (regex, *lowercasing*), *stemming* Sastrawi, dan penghapusan duplikasi. |
| `split_data.csv` | Rekapitulasi pembagian proporsi data (Training 80%, Validasi 10%, Testing 10%). |
| `classification_report.csv` | Laporan performa model per kelas sentimen (*precision*, *recall*, *f1-score*, *support*). |
| `summary_metrics.csv` | Ringkasan nilai metrik evaluasi model secara keseluruhan (*Accuracy*, rata-rata *F1-Score*, dsb). |
| `confusion_matrix.csv` | Matriks perbandingan data aktual vs data prediksi dalam format tabel CSV. |
| `evaluasi.csv` | Laporan evaluasi model lengkap yang ditransposisikan. |
| `prediksi_all.csv` | Seluruh data ulasan beserta hasil prediksi model, probabilitas keyakinan (*confidence*), dan status ambiguitas (*ambigu* / *tidak ambigu*). |
| `prediksi_all_spesifikasi.csv` | Data hasil penambahan label kategori aspek/spesifikasi ulasan (dihasilkan oleh script `spesifikasi.py`). |
| `ambigu_summary.csv` | Ringkasan jumlah data ulasan yang tergolong ambigu vs tidak ambigu. |
| `learning_curve.png` | Grafik kurva pembelajaran (*Accuracy* dan *Loss* per epoch) pada tahap pelatihan dan validasi. |
| `confusion_matrix.png` | Gambar diagram heatmap visualisasi *confusion matrix*. |
| `distribusi_label.png` | Diagram batang sebaran frekuensi kelas sentimen (*Positif*, *Netral*, *Negatif*). |
| `distribusi_ambigu.png` | Diagram batang persentase/jumlah data hasil deteksi ambiguitas. |

---

### 📂 Isi Folder `models/` (atau `model/`)
Folder `models/` digunakan untuk menyimpan berkas model pembelajaran mendalam hasil pelatihan:

| Nama File | Deskripsi / Penjelasan |
| :--- | :--- |
| `model.h5` | Model deep learning CNN terlatih yang tersimpan dalam format HDF5 Keras. Menyimpan konfigurasi arsitektur jaringan (*Embedding*, *Conv1D*, *MaxPooling1D*, *Dense*), bobot parameter (*weights*), serta status optimizer Adam yang dapat dimuat kembali (*load_model*) untuk melakukan inferensi ulasan baru tanpa perlu melatih ulang model. |

---

## ⚙️ Penjelasan Script Python

### 1. [main.py](main.py) — Pipeline Sentimen CNN Utama
Script utama yang menjalankan seluruh siklus machine learning end-to-end:
1. **Scraping Ulasan**: Mengambil ulasan game Roblox dari Google Play Store (`google_play_scraper`).
2. **Penerjemahan Teks**: Menerjemahkan ulasan ke Bahasa Indonesia (`googletrans`).
3. **Pembersihan & Preprocessing**: Menghapus URL, simbol asing, karakter non-alfanumerik, dan stemming kata dasar (`PySastrawi`).
4. **Pelabelan Sentimen**:
   - Rating 1–2: **Negatif**
   - Rating 3: **Netral**
   - Rating 4–5: **Positif**
5. **Pembagian Data**: Split dataset menjadi 80% data latih, 10% data validasi, dan 10% data uji.
6. **Pemodelan CNN 1D**: Arsitektur Text CNN dengan lapisan Embedding, Conv1D, MaxPooling, GlobalMaxPooling, Dense, Dropout (0.4), dan Dense Softmax.
7. **Evaluasi & Visualisasi**: Menghitung Classification Report, Confusion Matrix, dan mengekspor grafik kurva pembelajaran.
8. **Deteksi Ambiguitas**: Menandai ulasan sebagai **Ambigu** jika nilai *confidence* di bawah ambang batas (`< 0.5`) atau selisih probabilitas kelas tertinggi dengan posisi kedua terlalu dekat (`< 0.15`).
9. **Ekspor Model**: Menyimpan model ke `models/model.h5`.

---

### 2. [spesifikasi.py](spesifikasi.py) — Modul Kategorisasi Aspek / Spesifikasi Ulasan
Script ini digunakan untuk mengelompokkan ulasan pengguna ke dalam **kategori aspek masalah / fitur spesifik** (*Aspect-Based Categorization*) berdasarkan pencocokan kata kunci kontekstual menggunakan regex.

- **File Input:** `data/prediksi_all.csv` (dihasilkan oleh `main.py`).
- **File Output:** `data/prediksi_all_spesifikasi.csv` (menambahkan kolom baru `spesifikasi`).

#### Aspek yang Didukung & Contoh Kata Kunci:
1. **Edukasi & Batasan Usia**: `roblox kids`, `verifikasi umur`, `batas umur`, `anak`, `ortu`, `dewasa`, `remaja`, dll.
2. **Keamanan & Privasi**: `verifikasi wajah`, `scan wajah`, `keamanan akun`, `hack`, `hacker`, `cheater`, `cheat`, `phishing`, `aman`, dll.
3. **Fitur & Sosial (Chat/Komunikasi)**: `chat`, `ngobrol`, `komunikasi`, `mabar`, `teman`, `voice`, `mic`, `dm`, `add friend`, dll.
4. **Performa & Teknis (Bug/Error)**: `bug`, `error`, `lag`, `crash`, `force close`, `freeze`, `loading`, `gagal login`, `update`, `server`, `koneksi`, dll.
5. **Konten & Gameplay**: `map`, `game nya`, `grafis`, `level`, `tower`, `obby`, `item`, `skin`, `avatar`, `event`, dll.
6. **Umum/Lainnya**: Diberikan jika ulasan tidak mengandung kata kunci dari kelima aspek di atas.

> **Mekanisme Prioritas:** Pencocokan dilakukan berurutan dari aspek yang paling spesifik (*Edukasi & Batasan Usia* -> *Keamanan & Privasi* -> *Fitur & Sosial* -> *Performa & Teknis* -> *Konten & Gameplay*).

---

## 🚀 Cara Menjalankan Proyek

1. **Aktifkan Virtual Environment:**
   ```bash
   # Untuk Windows:
   venv\Scripts\activate
   
   # Untuk Linux/macOS:
   source venv/bin/activate
   ```

2. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Pipeline Utama (Training & Evaluasi):**
   ```bash
   python main.py
   ```
   *Langkah ini menghasilkan dataset olahan, model `models/model.h5`, grafik di folder `data/`, dan file `data/prediksi_all.csv`.*

4. **Jalankan Kategorisasi Aspek/Spesifikasi:**
   ```bash
   python spesifikasi.py
   ```
   *Langkah ini membaca `data/prediksi_all.csv` lalu mengekstrak aspek masalah/fitur ke file baru `data/prediksi_all_spesifikasi.csv`.*
