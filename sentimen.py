import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Sepak Bola Indonesia",
    page_icon="⚽"
)

# Load the saved model and vectorizer
model_filename = 'random_forest_model.pkl'  
vectorizer_filename = 'tfidf_vectorizer.pkl'  

with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(vectorizer_filename, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Negatif', 'Netral', 'Positif'])

# Buat daftar stop words bahasa Indonesia
factory = StopWordRemoverFactory()
stop_words = factory.get_stop_words()

# Load data
data = pd.read_csv('label_data_match.csv')
data.dropna(subset=['sentiment'], inplace=True)

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Pilih Halaman",
    ["Home Page", "Penjelasan Project", "Dashboard", "Analisis Tren Sentimen" , "Analisis Sentimen", "Analisis dengan Dataset", "Tweet Similarity Search"]
)

# Home Page
if page == "Home Page":
    st.title("Selamat Datang di Sistem Analisis Sentimen Sepak Bola Indonesia")

    # Menambahkan video
    video_path = "assets/bola.mp4"  
    try:
        st.video(video_path)
    except Exception as e:
        st.error("Tidak dapat memuat video. Silakan cek kembali file video.")
        st.write("Detail error:", str(e))

    st.write("""
    Tingkatkan pemahaman Anda tentang opini publik terhadap Sepak Bola Indonesia melalui sistem analisis sentimen kami. 
    Sistem kami memanfaatkan teknologi pembelajaran mesin untuk menganalisis data dari media sosial, 
    memberikan wawasan yang lebih dalam mengenai dukungan, kritik, dan harapan masyarakat terhadap Sepak Bola Indonesia 
    dalam setiap pertandingan kualifikasi Piala Dunia.
    """)

    st.subheader("Fitur:")
    st.write("""
    - **Analisis Sentimen:** Menilai apakah sentimen publik terhadap Sepak Bola Indonesia bersifat positif, negatif, atau netral.
    - **Tweet Similarity Search:** Temukan tweet dengan konten yang mirip, sehingga Anda bisa melihat tren dan diskusi yang berkembang.
    - **Unggah Data:** Menganalisis kumpulan data besar dengan mudah melalui file CSV yang sudah diproses.
    """)

    st.write("📊 **Jelajahi data, temukan pola, dan dukung Timnas Indonesia dengan wawasan berbasis data!**")
    

# Penjelasan Project
elif page == "Penjelasan Project":
    st.title("Penjelasan Project")

    # Tambahan Deskripsi
    st.subheader("Penjelasan Proyek")
    st.write("""
    ⚽️ **Proyek ini dirancang untuk menganalisis sentimen publik terhadap Sepak Bola Indonesia**, khususnya selama dan setelah pertandingan kualifikasi Piala Dunia. 
    Analisis ini dilakukan dengan memanfaatkan teknologi pembelajaran mesin yang mampu mengolah data dari platform X untuk mengungkap pola opini publik dalam bentuk sentimen positif 😊, negatif 😔, atau netral 😐.
    """)
    
    st.subheader("Latar Belakang")
    st.write("""
    📢 Setiap pertandingan Timnas Indonesia selalu memicu berbagai reaksi di media sosial. Pendapat-pendapat ini mencerminkan antusiasme 🎉, kritik 🧐, dan harapan 🙏 masyarakat terhadap performa tim. 
    Namun, tanpa analisis yang tepat, memahami tren sentimen dalam skala besar menjadi tantangan. Proyek ini hadir untuk memberikan solusi melalui analisis data secara mendalam dan terstruktur.
    """)
    
    st.subheader("Tujuan Utama")
    st.write("""
    1. 🔍 **Mengidentifikasi Sentimen:** Mengungkapkan bagaimana publik bereaksi terhadap pertandingan Sepak Bola Indonesia, baik dalam bentuk dukungan maupun kritik.
    2. 📈 **Menggambarkan Tren:** Memvisualisasikan perubahan sentimen dari waktu ke waktu untuk memahami pola opini masyarakat.
    3. 📊 **Menyediakan Wawasan Berbasis Data:** Mendukung pemilik kepentingan dalam mengambil keputusan strategis berdasarkan opini publik.
    """)

    st.subheader("Pendekatan")
    st.write("""
    - 📥 **Pengumpulan Data:** Mengambil data berupa tweet atau teks dari media sosial yang relevan dengan pertandingan Sepak Bola Indonesia.
    - 🧹 **Pre-processing Data:** Membersihkan data untuk menghilangkan informasi yang tidak relevan sehingga dapat dianalisis secara akurat.
    - 🤖 **Model Pembelajaran Mesin:** Menggunakan algoritma Random Forest untuk mengklasifikasikan sentimen ke dalam kategori positif 😊, negatif 😔, atau netral 😐.
    - 📊 **Evaluasi dan Visualisasi:** Menyediakan metrik kinerja model seperti akurasi dan F1-score serta visualisasi data sentimen dalam bentuk grafik 📉📈.
    """)

    st.subheader("Hasil yang Diharapkan")
    st.write("""
    ✨ Proyek ini diharapkan dapat menjadi alat yang berguna untuk memahami reaksi publik terhadap Sepak Bola Indonesia. 
    Dengan analisis berbasis data 📊, hasil proyek ini dapat digunakan oleh federasi sepak bola ⚽️, pelatih 🧑‍🏫, atau media 📰 untuk mengukur opini masyarakat secara real-time dan membuat keputusan yang lebih baik.
    """)

    st.markdown("""
    **"Menghubungkan Data dengan Harapan 💡, Mendukung Timnas dengan Wawasan 💪."**
    """)


# Dashboard
elif page == "Dashboard":
    st.title("Dashboard")
    st.write("Berikut adalah hasil analisis sentimen untuk beberapa pertandingan penting:")

    # Pertandingan 14: Indonesia vs Arab Saudi
    with st.expander("Indonesia vs Arab Saudi (19 November 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Arab Saudi - 19 November 2024")
        st.image("assets/1.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/2.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 13: Indonesia vs Jepang
    with st.expander("Indonesia vs Jepang (15 November 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Jepang - 15 November 2024")
        st.image("assets/3.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/4.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 12: China vs Indonesia
    with st.expander("China vs Indonesia (15 Oktober 2024)"):
        st.write("Analisis Sentimen untuk pertandingan China vs Indonesia - 15 Oktober 2024")
        st.image("assets/5.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/6.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 11: Bahrain vs Indonesia
    with st.expander("Bahrain vs Indonesia (10 Oktober 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Bahrain vs Indonesia - 10 Oktober 2024")
        st.image("assets/7.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/8.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 10: Indonesia vs Australia
    with st.expander("Indonesia vs Australia (10 September 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Australia - 10 September 2024")
        st.image("assets/9.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/10.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 9: Arab Saudi vs Indonesia 
    with st.expander("Arab Saudi vs Indonesia (6 September 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Arab Saudi vs Indonesia - 6 September 2024")
        st.image("assets/11.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/12.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 8: Indonesia vs Filipina
    with st.expander("Indonesia vs Filipina (11 Juni 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Filipina - 11 Juni 2024")
        st.image("assets/13.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/14.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 7: Indonesia vs Irak
    with st.expander("Indonesia vs Irak (6 Juni 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Irak - 6 Juni 2024")
        st.image("assets/15.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/16.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 6: Vietnam vs Indonesia
    with st.expander("Vietnam vs Indonesia (26 Maret 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Vietnam vs Indonesia - 26 Maret 2024")
        st.image("assets/17.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/18.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 5: Indonesia vs Vietnam
    with st.expander("Indonesia vs Vietnam (21 Maret 2024)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Vietnam - 21 Maret 2024")
        st.image("assets/19.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/20.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 4: Filipina vs Indonesia
    with st.expander("Filipina vs Indonesia (21 November 2023)"):
        st.write("Analisis Sentimen untuk pertandingan Filipina vs Indonesia - 21 November 2023")
        st.image("assets/21.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/22.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 3: Irak vs Indonesia 
    with st.expander("Irak vs Indonesia (16 November 2023)"):
        st.write("Analisis Sentimen untuk pertandingan Irak vs Indonesia - 16 November 2023")
        st.image("assets/23.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/24.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 2: Brunei vs Indonesia
    with st.expander("Brunei vs Indonesia (17 Oktober 2023)"):
        st.write("Analisis Sentimen untuk pertandingan Brunei vs Indonesia - 17 Oktober 2023")
        st.image("assets/25.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/26.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

    # Pertandingan 1: Indonesia vs Brunei
    with st.expander("Indonesia vs Brunei (12 Oktober 2023)"):
        st.write("Analisis Sentimen untuk pertandingan Indonesia vs Brunei - 12 Oktober 2023")
        st.image("assets/27.png", caption="Grafik Sentimen", use_column_width=True)
        st.image("assets/28.png", caption="Word Cloud dan Top 10 Kata Sentimen", use_column_width=True)

# Analisis Tren Sentimen
elif page == "Analisis Tren Sentimen":
    st.title("Analisis Tren Sentimen")
    st.write("""
    Halaman ini menyediakan analisis tren sentimen terhadap Sepak Bola Indonesia berdasarkan data media sosial. 
    Anda dapat melihat bagaimana sentimen publik berubah seiring waktu, menemukan pola musiman, dan menganalisis 
    dampak dari berbagai peristiwa pada opini publik.
    """)

    match_data = {
    "Pertandingan": [
        "Indonesia vs Brunei", "Brunei vs Indonesia", "Irak vs Indonesia", "Filipina vs Indonesia", 
        "Indonesia vs Vietnam", "Vietnam vs Indonesia", "Indonesia vs Irak", "Indonesia vs Filipina", 
        "Arab Saudi vs Indonesia", "Indonesia vs Australia", "Bahrain vs Indonesia", "China vs Indonesia", 
        "Indonesia vs Jepang", "Indonesia vs Arab Saudi"
    ],
    "Positif": [327, 236, 455, 501, 754, 943, 613, 1030, 661, 1035, 1517, 703, 675, 909],
    "Negatif": [348, 399, 1257, 716, 754, 1144, 963, 1002, 621, 993, 669, 710, 1084, 840],
    "Netral": [306, 307, 745, 524, 827, 852, 618, 883, 585, 859, 648, 1536, 676, 782]
    }

    # Membuat DataFrame dari data
    df_matches = pd.DataFrame(match_data)

    # Membuat line chart menggunakan Plotly
    fig = go.Figure()

    # Menambahkan garis untuk setiap sentimen dengan warna yang sesuai
    fig.add_trace(go.Scatter(
        x=df_matches["Pertandingan"],
        y=df_matches["Positif"],
        mode='lines+markers',
        name='Positif',
        line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=df_matches["Pertandingan"],
        y=df_matches["Negatif"],
        mode='lines+markers',
        name='Negatif',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=df_matches["Pertandingan"],
        y=df_matches["Netral"],
        mode='lines+markers',
        name='Netral',
        line=dict(color='blue')
    ))

    # Mengatur layout untuk mempercantik tampilan
    fig.update_layout(
        title="Visualisasi Sentimen Berdasarkan Pertandingan",
        xaxis_title="Pertandingan",
        yaxis_title="Jumlah Sentimen",
        legend_title="Kategori Sentimen",
        xaxis=dict(tickangle=45),
        template="plotly_white"
    )

    # Menampilkan chart di Streamlit
    st.plotly_chart(fig)


# Analisis Sentimen
elif page == "Analisis Sentimen":
    # Definisi kata kunci sentimen
    def get_sentiment_keywords():
        return {
            'positif': ['bagus', 'sangat', 'membanggakan', 'hebat', 'sukses', 'berhasil', 'menang', 'bangga', 'semangat', 'juara'],
            'negatif': ['buruk', 'mengecewakan', 'kalah', 'gagal', 'hancur', 'payah', 'jelek', 'parah', 'lemah', 'curang'],
            'netral': ['biasa', 'normal', 'standar', 'lumayan', 'cukup', 'seimbang', 'rata-rata']
        }

    st.title("Analisis Sentimen")
    
    # Tambahan Deskripsi
    st.subheader("Masukkan Kalimat, Temukan Sentimennya!")
    st.write("""
    Ingin tahu apakah sebuah kalimat mengandung sentimen positif, negatif, atau netral? 
    Masukkan teks Anda, dan sistem kami akan mengklasifikasikan sentimen berdasarkan analisis.
    
    📊 **Cek sentimen Anda dengan cepat dan mudah!**
    """)
    
    # Bagian Input dan Prediksi Tetap
    st.write("Masukkan kalimat untuk memprediksi sentimen:")


    # Input kalimat dari pengguna
    kalimat = st.text_input("Masukkan kalimat:")

    if kalimat:
        try:           
            kalimat_clean = kalimat.lower() 
            # Hitung skor sentimen menggunakan kata kunci yang telah didefinisikan            
            sentiment_keywords = get_sentiment_keywords()
            kata_positif = sentiment_keywords['positif']
            kata_negatif = sentiment_keywords['negatif']
            kata_netral = sentiment_keywords['netral']
              # Analisis sentimen berdasarkan kata kunci
            def hitung_skor_sentimen(teks, kata_kunci):
                return sum(1 for kata in kata_kunci if kata in teks)
            
            skor_positif = hitung_skor_sentimen(kalimat_clean, kata_positif)
            skor_negatif = hitung_skor_sentimen(kalimat_clean, kata_negatif)
            skor_netral = hitung_skor_sentimen(kalimat_clean, kata_netral)
            
            # Prediksi menggunakan model
            kalimat_vectorized = loaded_vectorizer.transform([kalimat]).toarray()           
            prediksi_model = loaded_model.predict(kalimat_vectorized)[0]
              # Tentukan sentimen berdasarkan gabungan rule-based dan model
            def tentukan_sentimen(skor_pos, skor_neg, skor_net, prediksi):
                if skor_net > 0 and skor_pos == 0 and skor_neg == 0:
                    return 'Netral' 
                elif skor_pos > skor_neg and skor_pos > skor_net:
                    return 'Positif' 
                elif skor_neg > skor_pos and skor_neg > skor_net:
                    return 'Negatif' 
                else:
                    return prediksi if isinstance(prediksi, str) else label_encoder.inverse_transform([prediksi])[0]
                    
            sentiment_label = tentukan_sentimen(skor_positif, skor_negatif, skor_netral, prediksi_model)

            # Tampilkan hasil dengan format yang lebih baik
            if sentiment_label == 'Positif':
                st.success(f"Sentimen prediksi: {sentiment_label} 😊")
            elif sentiment_label == 'Negatif':
                st.error(f"Sentimen prediksi: {sentiment_label} 😔")
            else:
                st.info(f"Sentimen prediksi: {sentiment_label} 😐")

        except NotFittedError:
            st.error("Vectorizer belum terlatih dengan benar.")

# Analisis dengan Dataset
elif page == "Analisis dengan Dataset":
    st.title("Analisis Sentimen dengan Dataset")
    
    # Tambahan Deskripsi
    st.subheader("Unggah Data Anda, Analisis Sentimen Seketika!")
    st.write("""
    Ingin menganalisis sentimen dari kumpulan data? Unggah file dalam format CSV dengan kolom berjudul **'text'** 
    yang sudah melalui proses **preprocessing**, dan sistem kami akan secara otomatis memproses serta mengklasifikasikan 
    setiap baris menjadi sentimen positif, negatif, atau netral.
    
    📊 **Mudah dan cepat untuk memahami pola sentimen dalam data Anda!** 
    Pastikan file Anda sudah siap dengan preprocessing untuk hasil terbaik. 
    Unggah file CSV yang sudah diproses untuk memprediksi sentimen:
    """)


    # Upload CSV file
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)

        # Pastikan kolom yang digunakan ada dalam data
        if 'text' not in df.columns:
            st.error("File CSV harus memiliki kolom 'text' untuk analisis sentimen.")
        else:
            # Menangani missing values (NaN)
            df['text'] = df['text'].fillna('')  # Mengganti NaN dengan string kosong

            # Prediksi sentimen untuk setiap kalimat
            try:
                # Cek apakah vectorizer sudah terlatih
                check_is_fitted(loaded_vectorizer, 'idf_')

                # Transformasi kalimat menggunakan vectorizer
                X = loaded_vectorizer.transform(df['text']).toarray()

                # Prediksi sentimen
                df['sentiment'] = loaded_model.predict(X)

                # Pemetaan label numerik ke label string
                sentiment_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

                # Pastikan bahwa prediksi model adalah numerik
                if df['sentiment'].dtype == 'int64':
                    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
                else:
                    df['sentiment_label'] = df['sentiment']

                # Tampilkan tabel lengkap
                st.write("Tabel Lengkap dengan Prediksi Sentimen:")
                st.dataframe(df[['text', 'sentiment_label']])

                # Visualisasi jumlah sentimen
                sentiment_counts = df['sentiment_label'].value_counts()
                st.write("Jumlah sentimen berdasarkan kategori:")
                st.bar_chart(sentiment_counts)

                # Menambahkan Word Cloud dan Top Kata per kategori sentimen
                sentiment_labels = ['Negatif', 'Netral', 'Positif']  # Sesuaikan dengan label yang digunakan dalam model

                # Membuat layout 3 kolom untuk menampilkan Word Cloud dan Top Kata
                col1, col2, col3 = st.columns(3)

                for idx, sentiment in enumerate(sentiment_labels):
                    # Filter data berdasarkan sentimen
                    sentiment_data = df[df['sentiment_label'] == sentiment]['text']

                    # Cek jika ada data untuk sentimen tersebut
                    if sentiment_data.empty:
                        st.write(f"Tidak ada data untuk Sentimen {sentiment}")
                    else:
                        # Gabungkan teks dari kategori sentimen tersebut
                        text_for_wordcloud = ' '.join(sentiment_data)

                        # Membuat WordCloud
                        wordcloud = WordCloud(width=400, height=400, background_color='white').generate(text_for_wordcloud)

                        # Menampilkan WordCloud di kolom yang sesuai
                        if idx == 0:
                            col1.subheader(f"Word Cloud untuk Sentimen {sentiment}")
                            col1.image(wordcloud.to_image(), use_column_width=True)
                        elif idx == 1:
                            col2.subheader(f"Word Cloud untuk Sentimen {sentiment}")
                            col2.image(wordcloud.to_image(), use_column_width=True)
                        else:
                            col3.subheader(f"Word Cloud untuk Sentimen {sentiment}")
                            col3.image(wordcloud.to_image(), use_column_width=True)

                        # Menampilkan Top 15 Kata dengan Frekuensi dalam bentuk tabel dengan indeks mulai dari 1
                        vectorizer = CountVectorizer(stop_words=stop_words)
                        X_sentiment = vectorizer.fit_transform(sentiment_data)
                        word_counts = pd.DataFrame({
                            'Kata': vectorizer.get_feature_names_out(),
                            'Frekuensi': X_sentiment.toarray().sum(axis=0)
                        }).sort_values(by='Frekuensi', ascending=False).head(15)

                        # Reset indeks agar mulai dari 1
                        word_counts.index = range(1, len(word_counts) + 1)

                        # Menampilkan tabel di kolom yang sesuai
                        if idx == 0:
                            col1.subheader(f"Top 15 Kata untuk Sentimen {sentiment}")
                            col1.table(word_counts)
                        elif idx == 1:
                            col2.subheader(f"Top 15 Kata untuk Sentimen {sentiment}")
                            col2.table(word_counts)
                        else:
                            col3.subheader(f"Top 15 Kata untuk Sentimen {sentiment}")
                            col3.table(word_counts)

            except NotFittedError:
                st.error("Vectorizer belum terlatih dengan benar.")

# Tweet Similarity Search
elif page == "Tweet Similarity Search":
    st.title('Pencarian Tweet Serupa')

    # Tambahan Deskripsi
    st.subheader("Cari Tweet Serupa dengan Cepat!")
    st.write("""
    Gunakan fitur **Pencarian Tweet Serupa** untuk menemukan tweet yang memiliki kemiripan konteks atau isi. 
    Cukup masukkan teks atau kata kunci, dan sistem kami akan menampilkan tweet-tweet lain yang relevan.
    
    🔍 **Temukan pola, diskusi, atau tren yang serupa dalam sekejap!**
    """)

    # Input pengguna dengan key unik
    user_input = st.text_input("Masukkan tweet untuk dibandingkan:", key="user_input")

    if user_input:
        # Transformasi tweet baru menggunakan vectorizer yang dimuat
        new_tweet_transformed = loaded_vectorizer.transform([user_input])

        # Transformasi semua tweet dalam dataset menggunakan vectorizer yang dimuat
        X_transformed = loaded_vectorizer.transform(data['tweet_after_prepros'])

        # Hitung cosine similarity antara tweet baru dan semua tweet dalam dataset
        similarity_scores = cosine_similarity(new_tweet_transformed, X_transformed)

        # Ambil 5 tweet paling serupa
        top_n = 10
        top_n_indices = similarity_scores.argsort()[0][-top_n:][::-1]

        # Persiapkan dataframe untuk tweet serupa
        similar_tweets = data.iloc[top_n_indices][['tweet_after_prepros']].copy()
        similar_tweets['Similarity Score'] = similarity_scores[0][top_n_indices]

        # Tampilkan hasil dalam bentuk tabel
        st.subheader("10 Tweet Serupa Teratas")
        st.write(similar_tweets.rename(columns={'tweet_after_prepros': 'Tweet'}).reset_index(drop=True))

