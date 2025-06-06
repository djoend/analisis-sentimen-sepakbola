# Analisis Sentimen Sepak Bola Indonesia 🇮🇩⚽

Sistem analisis sentimen yang dirancang untuk menganalisis opini publik terhadap Sepak Bola Indonesia, khususnya selama pertandingan kualifikasi Piala Dunia.

## 🌟 Fitur Utama

- **Analisis Sentimen Real-time**: Analisis teks untuk menentukan sentimen (positif, negatif, atau netral)
- **Dashboard Interaktif**: Visualisasi data sentimen dari berbagai pertandingan
- **Tweet Similarity Search**: Pencarian tweet dengan konten yang mirip
- **Analisis Dataset**: Kemampuan untuk menganalisis dataset dalam format CSV
- **Visualisasi Data**: Word clouds dan grafik distribusi sentimen

## 🛠️ Teknologi yang Digunakan

- Python 3.8+
- Streamlit
- Scikit-learn
- NLTK
- Sastrawi (Indonesian NLP Library)
- Pandas
- NumPy
- Plotly
- WordCloud

## 📋 Prasyarat

- Python 3.8 atau versi yang lebih baru
- pip (Python package installer)

## 🚀 Cara Menjalankan

1. Clone repository ini:
```bash
git clone [URL_REPOSITORY]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run sentimen.py
```

## 📊 Struktur Proyek

```
├── assets/                 # Gambar dan video untuk dashboard
├── sentimen.py            # File utama aplikasi Streamlit
├── requirements.txt       # Daftar package yang diperlukan
├── random_forest_model.pkl # Model machine learning yang sudah dilatih
└── tfidf_vectorizer.pkl   # Vectorizer yang sudah dilatih
```

## 📱 Halaman Aplikasi

1. **Home Page**: Halaman utama dengan video intro dan penjelasan sistem
2. **Penjelasan Project**: Detail tentang latar belakang dan tujuan project
3. **Dashboard**: Visualisasi hasil analisis sentimen dari berbagai pertandingan
4. **Analisis Tren Sentimen**: Grafik tren sentimen sepanjang waktu
5. **Analisis Sentimen**: Tool untuk menganalisis sentimen teks
6. **Analisis dengan Dataset**: Analisis sentimen untuk dataset CSV
7. **Tweet Similarity Search**: Pencarian tweet yang mirip

## 📚 Model dan Dataset

- Model menggunakan Random Forest Classifier
- Data diambil dari platform X (Twitter)
- Preprocessing menggunakan Sastrawi untuk Bahasa Indonesia
- Model dilatih dengan dataset yang sudah dilabel manual

## 🤝 Kontribusi

Kontribusi selalu diterima! Silakan buat pull request atau laporkan issues jika menemukan bug.

## 📜 Lisensi

[MIT License](LICENSE)
