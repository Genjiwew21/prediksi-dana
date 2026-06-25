"""
app.py
=======
Aplikasi Streamlit untuk Analisis Sentimen Ulasan Pengguna
Aplikasi DANA di Google Play Store.

Menggunakan model yang sudah dilatih sebelumnya (lihat train_model.py)
dan disimpan di folder models/ dalam format pickle, sehingga aplikasi
ini akan terbuka dengan cepat tanpa perlu training ulang.

Jalankan dengan:
    streamlit run app.py
"""

import os
import pickle
import re
import string

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import zipfile

# --- TRICK KHUSUS UNTUK EKSTRAK MODEL ZIP ---
if not os.path.exists('models') and os.path.exists('models.zip'):
    with zipfile.ZipFile('models.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
# ──────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen DANA",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"

COLOR_MAP = {"positif": "#2ecc71", "netral": "#95a5a6", "negatif": "#e74c3c"}
LABEL_ORDER = ["positif", "netral", "negatif"]


# ──────────────────────────────────────────────────────────────────────────
# UTIL: PREPROCESSING (harus identik dengan train_model.py)
# ──────────────────────────────────────────────────────────────────────────
def cleaning_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def get_stemmer_stopwords():
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    return stemmer, stopwords


def preprocess_text(text):
    """Pipeline preprocessing lengkap untuk satu kalimat input user."""
    stemmer, stopwords = get_stemmer_stopwords()
    text = cleaning_text(text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────────────────
# LOAD ARTEFAK MODEL (dengan cache supaya tidak load ulang setiap interaksi)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    required_files = [
        "tfidf_vectorizer.pkl",
        "model_naive_bayes.pkl",
        "model_svm.pkl",
        "model_random_forest.pkl",
        "metrics.pkl",
        "confusion_matrices.pkl",
    ]
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, missing

    artifacts = {}
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        artifacts["tfidf"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model_naive_bayes.pkl"), "rb") as f:
        artifacts["nb"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model_svm.pkl"), "rb") as f:
        artifacts["svm"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model_random_forest.pkl"), "rb") as f:
        artifacts["rf"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "metrics.pkl"), "rb") as f:
        artifacts["metrics"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "confusion_matrices.pkl"), "rb") as f:
        artifacts["conf_matrices"] = pickle.load(f)

    sample_path = os.path.join(MODELS_DIR, "sample_data.pkl")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            artifacts["sample_data"] = pickle.load(f)
    else:
        artifacts["sample_data"] = None

    return artifacts, []


artifacts, missing_files = load_artifacts()

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("📱 Analisis Sentimen DANA")
st.sidebar.markdown(
    "Dashboard interaktif analisis sentimen ulasan pengguna aplikasi "
    "**DANA** di Google Play Store, menggunakan algoritma "
    "**Naïve Bayes**, **SVM**, dan **Random Forest**."
)
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Ringkasan", "📊 Eksplorasi Data", "🤖 Perbandingan Model", "🔮 Coba Prediksi Sendiri"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan Streamlit · Model: TF-IDF + ML klasik")

# ──────────────────────────────────────────────────────────────────────────
# JIKA MODEL BELUM ADA -> TAMPILKAN PETUNJUK, BUKAN ERROR MENTAH
# ──────────────────────────────────────────────────────────────────────────
if artifacts is None:
    st.title("📱 Analisis Sentimen Ulasan Pengguna Aplikasi DANA")
    st.error("⚠️ Model belum ditemukan di folder `models/`.")
    st.markdown(
        f"""
        File yang belum ada: `{'`, `'.join(missing_files)}`

        **Langkah agar aplikasi bisa berjalan:**

        1. Pastikan file `data_dana.csv` ada di folder yang sama dengan `app.py`.
        2. Jalankan script training **satu kali saja** dari terminal:
           ```bash
           python train_model.py
           ```
        3. Tunggu sampai proses selesai (akan muncul folder `models/`).
        4. Jalankan ulang aplikasi:
           ```bash
           streamlit run app.py
           ```
        """
    )
    st.stop()

metrics_data = artifacts["metrics"]
df_metrics = metrics_data["df_metrics"]
label_dist = metrics_data["label_distribution"]
conf_matrices = artifacts["conf_matrices"]
labels_cm = metrics_data["labels_cm"]
reports = metrics_data["reports"]

MODEL_OBJ = {
    "Naive Bayes": artifacts["nb"],
    "SVM": artifacts["svm"],
    "Random Forest": artifacts["rf"],
}

# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 1: RINGKASAN
# ──────────────────────────────────────────────────────────────────────────
if page == "🏠 Ringkasan":
    st.title("📱 Analisis Sentimen Ulasan Pengguna Aplikasi DANA")
    st.markdown(
        "##### Klasifikasi sentimen ulasan Google Play Store menggunakan "
        "**Naïve Bayes**, **SVM**, dan **Random Forest**"
    )
    st.markdown("---")

    total = metrics_data["n_total"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Bersih", f"{total:,}")
    c2.metric("Data Training", f"{metrics_data['n_train']:,}")
    c3.metric("Data Testing", f"{metrics_data['n_test']:,}")
    best_model = df_metrics["F1-Score"].idxmax()
    c4.metric("Model Terbaik (F1)", best_model,
              f"{df_metrics.loc[best_model, 'F1-Score']*100:.2f}%")

    st.markdown("### 🏷️ Distribusi Label Sentimen")
    col1, col2 = st.columns([1, 1])

    with col1:
        dist_df = pd.DataFrame({
            "Sentimen": list(label_dist.keys()),
            "Jumlah": list(label_dist.values()),
        })
        dist_df["Sentimen"] = dist_df["Sentimen"].str.capitalize()
        st.bar_chart(dist_df.set_index("Sentimen"), color="#3498db")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ordered_labels = [l for l in LABEL_ORDER if l in label_dist]
        sizes = [label_dist[l] for l in ordered_labels]
        colors = [COLOR_MAP[l] for l in ordered_labels]
        ax.pie(
            sizes, labels=[l.capitalize() for l in ordered_labels],
            autopct="%1.1f%%", colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title("Proporsi Sentimen", fontweight="bold")
        st.pyplot(fig, use_container_width=True)

    st.info(
        "💡 **Catatan:** Kelas *netral* memiliki proporsi data yang jauh lebih "
        "kecil dibanding *positif* dan *negatif* (class imbalance), sehingga "
        "wajar jika performa model pada kelas ini relatif lebih rendah."
    )

    st.markdown("### 🏆 Ringkasan Performa Model")
    df_display = (df_metrics * 100).round(2)
    st.dataframe(df_display.style.highlight_max(axis=0, color="#d4f7dc"),
                 use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 2: EKSPLORASI DATA
# ──────────────────────────────────────────────────────────────────────────
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")

    sample_data = artifacts["sample_data"]
    if sample_data is None:
        st.warning("Data sample tidak tersedia. Jalankan ulang `train_model.py` "
                    "untuk menghasilkan file `sample_data.pkl`.")
    else:
        st.markdown(f"Menampilkan contoh acak dari **{len(sample_data):,}** "
                     f"baris data (dari total data yang sudah diproses).")

        sentimen_filter = st.multiselect(
            "Filter berdasarkan sentimen:",
            options=["positif", "netral", "negatif"],
            default=["positif", "netral", "negatif"],
        )
        filtered = sample_data[sample_data["labeling"].isin(sentimen_filter)]
        st.dataframe(
            filtered[["Ulasan", "Rating", "labeling", "final_text"]]
            .rename(columns={
                "Ulasan": "Ulasan Asli", "labeling": "Sentimen",
                "final_text": "Teks Setelah Preprocessing",
            }),
            use_container_width=True, height=400,
        )

        st.markdown("### ☁️ Word Cloud per Sentimen")
        try:
            from wordcloud import WordCloud
            cols = st.columns(3)
            wc_colors = {"positif": "Greens", "negatif": "Reds", "netral": "Blues"}
            for col, sent in zip(cols, ["positif", "negatif", "netral"]):
                with col:
                    st.markdown(f"**{sent.capitalize()}**")
                    teks = " ".join(
                        sample_data[sample_data["labeling"] == sent]["final_text"].dropna()
                    )
                    if teks.strip():
                        wc = WordCloud(
                            width=400, height=300, background_color="white",
                            colormap=wc_colors[sent], max_words=80,
                        ).generate(teks)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.caption("Tidak ada data cukup untuk kategori ini.")
        except ImportError:
            st.warning("Package `wordcloud` belum terinstal. "
                        "Tambahkan `wordcloud` ke requirements.txt.")

# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 3: PERBANDINGAN MODEL
# ──────────────────────────────────────────────────────────────────────────
elif page == "🤖 Perbandingan Model":
    st.title("🤖 Perbandingan Model")

    st.markdown("### 📈 Metrik Evaluasi")
    df_display = (df_metrics * 100).round(2)
    st.dataframe(
        df_display.style.highlight_max(axis=0, color="#d4f7dc"),
        use_container_width=True,
    )

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    chart_df = df_display[metric_cols]
    st.bar_chart(chart_df.T)

    st.markdown("---")
    st.markdown("### 🧮 Confusion Matrix per Model")
    selected_model = st.selectbox("Pilih model:", list(MODEL_OBJ.keys()))

    cm = conf_matrices[selected_model]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {selected_model}", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, use_container_width=False)

    st.markdown("### 📋 Classification Report")
    st.code(reports[selected_model], language=None)

# ──────────────────────────────────────────────────────────────────────────
# HALAMAN 4: PREDIKSI MANUAL
# ──────────────────────────────────────────────────────────────────────────
elif page == "🔮 Coba Prediksi Sendiri":
    st.title("🔮 Coba Prediksi Sentimen Sendiri")
    st.markdown(
        "Masukkan contoh ulasan (boleh Bahasa Indonesia informal) lalu pilih "
        "model untuk melihat hasil prediksi sentimennya secara langsung."
    )

    user_text = st.text_area(
        "✍️ Tulis ulasan di sini:",
        placeholder="Contoh: Aplikasinya sangat membantu untuk transfer dan bayar tagihan, mantap!",
        height=120,
    )
    model_choice = st.selectbox(
        "Pilih model untuk prediksi:",
        list(MODEL_OBJ.keys()),
    )

    if st.button("🔍 Prediksi Sentimen", type="primary"):
        if not user_text.strip():
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses teks dan melakukan prediksi..."):
                processed = preprocess_text(user_text)
                vector = artifacts["tfidf"].transform([processed])
                model = MODEL_OBJ[model_choice]
                pred = model.predict(vector)[0]

                # Skor keyakinan jika tersedia (NB & RF punya predict_proba, SVM tidak)
                proba_info = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vector)[0]
                    classes = model.classes_
                    proba_info = dict(zip(classes, proba))

            emoji_map = {"positif": "😊", "netral": "😐", "negatif": "😡"}
            color = COLOR_MAP.get(pred, "#333")
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:10px; background-color:{color}22;
                border:2px solid {color}; text-align:center;">
                    <h2 style="color:{color}; margin:0;">
                        {emoji_map.get(pred, '')} Sentimen: {pred.upper()}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔬 Lihat detail hasil preprocessing"):
                st.write("**Teks setelah preprocessing:**")
                st.code(processed if processed.strip() else "(kosong setelah preprocessing)")

            if proba_info:
                st.markdown("**Tingkat keyakinan model:**")
                proba_df = pd.DataFrame({
                    "Sentimen": [k.capitalize() for k in proba_info.keys()],
                    "Probabilitas (%)": [round(v * 100, 2) for v in proba_info.values()],
                })
                st.bar_chart(proba_df.set_index("Sentimen"))
            elif model_choice == "SVM":
                st.caption(
                    "ℹ️ Model SVM (LinearSVC) tidak menyediakan skor probabilitas, "
                    "hanya label prediksi langsung."
                )

    st.markdown("---")
    st.caption(
        "⚠️ Model ini dilatih hanya dari ulasan aplikasi DANA, sehingga hasil "
        "prediksi paling akurat untuk konteks ulasan aplikasi serupa."
    )
