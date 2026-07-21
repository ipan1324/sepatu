# Implementasi dan Analisis Perbandingan Transfer Learning MobileNetV2 dan EfficientNetB0 untuk Klasifikasi Jenis Sepatu Berbasis Web Menggunakan Flask

Proyek ini adalah Capstone Project untuk Mata Kuliah Kecerdasan Buatan (Artificial Intelligence) dengan fokus pada Computer Vision. Proyek ini membandingkan kinerja dua arsitektur *Transfer Learning*, yaitu **MobileNetV2** dan **EfficientNetB0**, dalam mengklasifikasikan jenis sepatu. Model terbaik kemudian di-deploy menggunakan Flask ke dalam sebuah antarmuka web modern berbasis Bootstrap 5.

## Fitur Utama
- **Transfer Learning**: Membandingkan MobileNetV2 dan EfficientNetB0 menggunakan weights dari ImageNet.
- **Data Augmentation**: Menggunakan teknik augmentasi gambar (rotasi, flip, zoom, shift) untuk meningkatkan ketahanan model.
- **Visualisasi Performa**: Menampilkan grafik akurasi, *loss*, dan *Confusion Matrix*.
- **Web Application**: Aplikasi web dinamis dengan fitur *upload* gambar, *loading state*, dan *confidence score*.
- **UI Modern**: Desain responsif, modern, dan dilengkapi efek *glassmorphism* serta animasi menggunakan Bootstrap 5 dan Vanilla CSS.

## Struktur Direktori
```
project/
├── dataset/                # Folder berisi dataset gambar sepatu terbagi per kelas
├── models/                 # Tempat menyimpan model terlatih (.keras/.h5)
├── static/                 
│   ├── css/style.css       # File stylesheet kustom
│   └── js/script.js        # File script JS (loading, preview gambar)
├── templates/              
│   ├── index.html          # Halaman beranda
│   ├── about.html          # Halaman tentang proyek
│   ├── predict.html        # Halaman prediksi/upload
│   └── result.html         # Halaman hasil prediksi
├── uploads/                # Folder penyimpanan sementara gambar yang diunggah
├── app.py                  # Aplikasi utama Flask
├── compare_models.py       # Script untuk membandingkan model
├── predict.py              # Script modul pembantu untuk inferensi model
├── train_efficientnet.py   # Script pelatihan model EfficientNetB0
├── train_mobilenet.py      # Script pelatihan model MobileNetV2
└── requirements.txt        # Daftar dependensi proyek
```

## Dataset
Pastikan dataset gambar sepatu Anda diletakkan di dalam folder `dataset/` (atau `Shoes Dataset/` dan diatur dalam script *training*), terbagi dalam beberapa subfolder berdasarkan nama kelas.
Contoh:
```
dataset/
├── sneakers/
├── boots/
├── sandals/
└── ...
```

## Persyaratan (Requirements)
Proyek ini membutuhkan Python 3.11. Anda bisa menginstal semua modul yang dibutuhkan melalui `requirements.txt`.

### Cara Install
1. *Clone* atau unduh repository ini.
2. Buka terminal/Command Prompt di dalam folder proyek.
3. Buat dan aktifkan *Virtual Environment* (opsional tapi disarankan):
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/Mac:
   source venv/bin/activate
   ```
4. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Pelatihan (Training)
Untuk melatih model, jalankan script yang diinginkan secara bergiliran. Script ini akan memproses dataset, melatih model, dan menyimpan hasil evaluasi beserta file model ke direktori saat ini / `models/`.

1. **Pelatihan MobileNetV2**
   ```bash
   python train_mobilenet.py
   ```
2. **Pelatihan EfficientNetB0**
   ```bash
   python train_efficientnet.py
   ```
3. **Membandingkan Model**
   Pastikan Anda sudah menjalankan kedua pelatihan di atas, lalu jalankan:
   ```bash
   python compare_models.py
   ```

## Cara Menjalankan Aplikasi Web Flask
1. Pastikan Anda telah memiliki model hasil pelatihan (misalnya `models/mobilenetv2_model.keras`) yang tersimpan di dalam folder `models/`. (Anda bisa menyesuaikan nama file di `app.py`).
2. Jalankan aplikasi Flask:
   ```bash
   python app.py
   ```
3. Buka browser dan akses alamat yang diberikan (biasanya `http://127.0.0.1:5000/`).

## Hasil Analisis dan Kesimpulan
Berdasarkan hasil *training* (lihat keluaran `compare_models.py`), kita membandingkan:
- **Ukuran dan Parameter Model**: MobileNetV2 biasanya memiliki jumlah parameter yang lebih sedikit sehingga ukuran modelnya lebih kecil dibandingkan varian EfficientNet, namun EfficientNetB0 telah dioptimalkan secara arsitektur melalui *compound scaling* yang memberikan efisiensi luar biasa.
- **Kecepatan Training**: MobileNetV2 lebih cepat untuk proses pelatihan per *epoch*-nya.
- **Akurasi, Presisi, dan Recall**: EfficientNetB0 seringkali memberikan akurasi dan ketepatan prediksi (*F1-Score*) yang lebih tinggi pada dataset kompleks karena kapasitas ekstraksi fiturnya yang lebih mendalam.

**Kesimpulan Utama**: MobileNetV2 sangat cocok jika aplikasi membutuhkan inferensi cepat di lingkungan komputasi terbatas, sedangkan EfficientNetB0 direkomendasikan jika tujuan utamanya adalah maksimalisasi akurasi klasifikasi.

## Tangkapan Layar (Screenshot)
*(Tambahkan gambar screenshot aplikasi web di sini)*
- Home Page
- Upload Page
- Result Page
