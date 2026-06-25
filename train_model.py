"""
train_model.py
================
Script ini menjalankan ULANG pipeline dari notebook
`Analisis_Sentimen_DANA_Final_Clean.ipynb` secara end-to-end:

  1. Load dataset data_dana.csv
  2. Labeling sentimen berdasarkan Rating
  3. Text preprocessing (Cleaning -> Case Folding -> Tokenizing
     -> Stopword Removal -> Stemming)
  4. TF-IDF Vectorization
  5. Training 3 model: Naive Bayes, SVM, Random Forest
  6. Menyimpan vectorizer + model + metrik ke folder models/
     dalam bentuk file .pkl, supaya bisa langsung dipakai
     oleh aplikasi Streamlit (app.py) tanpa training ulang.

Cara menjalankan (sekali saja, sebelum deploy):
    python train_model.py

Setelah selesai, akan muncul folder `models/` berisi:
    - tfidf_vectorizer.pkl
    - model_naive_bayes.pkl
    - model_svm.pkl
    - model_random_forest.pkl
    - metrics.pkl
    - confusion_matrices.pkl
"""

import os
import re
import string
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

DATA_PATH = "data_dana.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ──────────────────────────────────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────────────────────────────────
log("Membaca dataset ...")
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    on_bad_lines="skip",
    low_memory=False,
    usecols=["Nama User", "Ulasan", "Rating", "Tanggal", "Likes", "Versi App"],
    dtype={
        "Nama User": str, "Ulasan": str, "Rating": str,
        "Tanggal": str, "Likes": str, "Versi App": str,
    },
)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce")
df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y %H.%M", errors="coerce")
df = df.dropna(subset=["Ulasan", "Rating"]).reset_index(drop=True)
log(f"Dataset dimuat: {len(df):,} baris")

# ──────────────────────────────────────────────────────────────────────────
# 2. Labeling Sentimen Berdasarkan Rating
# ──────────────────────────────────────────────────────────────────────────
conditions = [
    df["Rating"].isin([1.0, 2.0]),
    df["Rating"] == 3.0,
    df["Rating"].isin([4.0, 5.0]),
]
choices = ["negatif", "netral", "positif"]
df["labeling"] = np.select(conditions, choices, default="unknown")
df = df[df["labeling"] != "unknown"].reset_index(drop=True)
log("Distribusi label:")
log(df["labeling"].value_counts().to_dict())

# ──────────────────────────────────────────────────────────────────────────
# 3. Text Preprocessing
# ──────────────────────────────────────────────────────────────────────────
log("Preprocessing teks (cleaning -> case folding -> tokenizing "
    "-> stopword removal -> stemming) ...")


def cleaning_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["cleaning"] = df["Ulasan"].apply(cleaning_text)
df["case_folding"] = df["cleaning"].str.lower()
df["tokenizing"] = df["case_folding"].apply(lambda x: x.split())

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())


def remove_stopwords(tokens):
    return [w for w in tokens if w not in stopwords]


df["stopword_removal"] = df["tokenizing"].apply(remove_stopwords)

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()


def stemming(tokens):
    return [stemmer.stem(w) for w in tokens]


log("Stemming (proses ini paling lama, mohon tunggu) ...")
df["stemming"] = df["stopword_removal"].apply(stemming)
df["final_text"] = df["stemming"].apply(lambda x: " ".join(x))
df = df[df["final_text"].str.strip() != ""].reset_index(drop=True)
log(f"Total baris siap diproses: {len(df):,}")

# ──────────────────────────────────────────────────────────────────────────
# 4. TF-IDF Vectorization & Split Data
# ──────────────────────────────────────────────────────────────────────────
log("TF-IDF vectorization & split data ...")
X = df["final_text"]
y = df["labeling"]

tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
log(f"Data training : {X_train.shape[0]:,} sampel")
log(f"Data testing  : {X_test.shape[0]:,} sampel")

# ──────────────────────────────────────────────────────────────────────────
# 5. Training & Evaluasi 3 Model
# ──────────────────────────────────────────────────────────────────────────
log("Training Naive Bayes (ComplementNB) ...")
nb_model = ComplementNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

log("Training SVM (LinearSVC) ...")
svm_model = LinearSVC(max_iter=3000, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

log("Training Random Forest ...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ──────────────────────────────────────────────────────────────────────────
# 6. Hitung Metrik & Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────
log("Menghitung metrik evaluasi ...")
model_names = ["Naive Bayes", "SVM", "Random Forest"]
y_preds = [y_pred_nb, y_pred_svm, y_pred_rf]
labels_cm = ["negatif", "netral", "positif"]

metrics = {}
conf_matrices = {}
reports = {}
for name, yp in zip(model_names, y_preds):
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, yp),
        "Balanced Accuracy": balanced_accuracy_score(y_test, yp),
        "Precision": precision_score(y_test, yp, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, yp, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, yp, average="weighted", zero_division=0),
    }
    conf_matrices[name] = confusion_matrix(y_test, yp, labels=labels_cm)
    reports[name] = classification_report(
        y_test, yp, labels=["positif", "netral", "negatif"], zero_division=0
    )
    log(f"{name}: Accuracy={metrics[name]['Accuracy']:.4f}, "
        f"F1={metrics[name]['F1-Score']:.4f}")

df_metrics = pd.DataFrame(metrics).T
df_metrics.index.name = "Model"

label_dist = df["labeling"].value_counts().to_dict()

# ──────────────────────────────────────────────────────────────────────────
# 7. Simpan Semua Artefak ke models/
# ──────────────────────────────────────────────────────────────────────────
log("Menyimpan model & artefak ke folder models/ ...")
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(MODELS_DIR, "model_naive_bayes.pkl"), "wb") as f:
    pickle.dump(nb_model, f)

with open(os.path.join(MODELS_DIR, "model_svm.pkl"), "wb") as f:
    pickle.dump(svm_model, f)

with open(os.path.join(MODELS_DIR, "model_random_forest.pkl"), "wb") as f:
    pickle.dump(rf_model, f)

with open(os.path.join(MODELS_DIR, "metrics.pkl"), "wb") as f:
    pickle.dump(
        {
            "df_metrics": df_metrics,
            "label_distribution": label_dist,
            "labels_cm": labels_cm,
            "reports": reports,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "n_total": len(df),
        },
        f,
    )

with open(os.path.join(MODELS_DIR, "confusion_matrices.pkl"), "wb") as f:
    pickle.dump(conf_matrices, f)

# Simpan sample data hasil preprocessing untuk ditampilkan di dashboard (opsional, dibatasi)
df_sample = df[["Ulasan", "Rating", "labeling", "final_text"]].sample(
    n=min(2000, len(df)), random_state=42
).reset_index(drop=True)
with open(os.path.join(MODELS_DIR, "sample_data.pkl"), "wb") as f:
    pickle.dump(df_sample, f)

log("✅ Selesai! Semua model & artefak tersimpan di folder 'models/'.")
log("Sekarang jalankan: streamlit run app.py")
