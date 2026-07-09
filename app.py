import re
import os

import joblib
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Sentimen - Livin' by Mandiri",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------
# STOPWORDS (cache)
# ---------------------------------------------------------
@st.cache_resource
def load_stopwords():
    try:
        stop_words = nltk.corpus.stopwords.words("indonesian")
    except LookupError:
        nltk.download("stopwords")
        stop_words = nltk.corpus.stopwords.words("indonesian")
    return set(stop_words)


STOP_WORDS = load_stopwords()

KAMUS_TIDAK_BAKU = {
    "apk": "aplikasi",
    "app": "aplikasi",
    "ngedit": "edit",
    "atm": "automated teller machine",
}

HAPUS_KATA = ["ya", "sih", "sok", "for", "nih", "is", "ok", "gpt", "a", "oke"]


# ---------------------------------------------------------
# PREPROCESSING (sama seperti notebook)
# ---------------------------------------------------------
def remove_url(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U0001F004-\U0001F0CF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_symbols(text): 
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_numbers(text):
    return re.sub(r"\d", "", text)


def cleaning(text):
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_symbols(text)
    text = remove_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def normalisasi(text):
    words = text.split()
    replaced = []
    for word in words:
        if word in KAMUS_TIDAK_BAKU:
            baku = KAMUS_TIDAK_BAKU[word]
            if isinstance(baku, str) and all(c.isalpha() or c.isspace() for c in baku):
                replaced.append(baku)
        else:
            replaced.append(word)
    return " ".join(replaced)


def preprocess_text(raw_text: str) -> str:
    text = cleaning(raw_text)
    text = text.lower()
    text = normalisasi(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    text = " ".join(tokens)
    text = " ".join([w for w in text.split() if w not in HAPUS_KATA])
    return text


# ---------------------------------------------------------
# LOAD MODEL & VECTORIZER
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("model_nb.pkl")
    return vectorizer, model


vectorizer, model = load_model_and_vectorizer()


# ---------------------------------------------------------
# LOAD DATASET (opsional, untuk dashboard statistik)
# ---------------------------------------------------------
DATA_CANDIDATES = [
    "Hasil_Labelling_Data_2Class.csv",
    "Hasil_Preprocessing_Data.csv",
    "dataset_livin.csv",
]


@st.cache_data
def load_dataset():
    errors = []
    for fname in DATA_CANDIDATES:
        if os.path.exists(fname):
            for enc in ["utf-8", "utf-8-sig", "latin1"]:
                try:
                    df = pd.read_csv(fname, encoding=enc)
                    return df, fname, errors
                except Exception as e:
                    errors.append(f"{fname} ({enc}): {e}")
                    continue
    return None, None, errors


df, source_file, load_errors = load_dataset()

# ---------------------------------------------------------
# SIDEBAR NAVIGASI
# ---------------------------------------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih halaman",
    ["Dashboard", "Prediksi Sentimen"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Model: Multinomial Naive Bayes")
st.sidebar.caption("Dataset: Livin' by Mandiri 2025")
if source_file:
    st.sidebar.success(f"Data terbaca dari: {source_file}")
else:
    st.sidebar.warning("File dataset CSV tidak ditemukan di folder ini. Halaman Dashboard akan terbatas.")
    with st.sidebar.expander("Debug: file di folder ini"):
        st.write(f"Working dir: {os.getcwd()}")
        st.write([f for f in os.listdir(".") if f.endswith(".csv")])

# ===========================================================
# HALAMAN 1 — DASHBOARD
# ===========================================================
if page == "Dashboard":
    st.title("Dashboard Analisis Sentimen Ulasan pengguna Aplikasi Livin' by Mandiri")
    st.caption("Tahun 2025")

    if df is None:
        st.info(
            "Untuk menampilkan dashboard statistik, taruh salah satu file berikut "
            "di folder yang sama dengan app.py: "
            + ", ".join(f"`{f}`" for f in DATA_CANDIDATES)
        )
        if load_errors:
            with st.expander("Lihat detail error (untuk debugging)"):
                for err in load_errors:
                    st.code(err)
    else:
        # Cari kolom sentimen / score secara fleksibel
        sentiment_col = None
        for c in ["Sentiment", "sentiment", "label"]:
            if c in df.columns:
                sentiment_col = c
                break

        score_col = "score" if "score" in df.columns else None

        # Kalau belum ada kolom Sentiment tapi ada score, buat label otomatis
        if sentiment_col is None and score_col is not None:
            def label_sentimen(score):
                if score in [1, 2]:
                    return "Negatif"
                elif score in [3, 4, 5]:
                    return "Positif"
                return None

            df["Sentiment"] = df[score_col].apply(label_sentimen)
            sentiment_col = "Sentiment"

        col1, col2, col3, col4 = st.columns(4)

        total_data = len(df)
        col1.metric("Total Ulasan", f"{total_data:,}")

        if sentiment_col is not None:
            counts = df[sentiment_col].value_counts()
            pos = counts.get("Positif", 0)
            neg = counts.get("Negatif", 0)
            total_valid = pos + neg if (pos + neg) > 0 else 1

            col2.metric("Sentimen Positif", f"{pos:,}", f"{pos / total_valid * 100:.1f}%")
            col3.metric("Sentimen Negatif", f"{neg:,}", f"{neg / total_valid * 100:.1f}%")
        else:
            col2.metric("Sentimen Positif", "-")
            col3.metric("Sentimen Negatif", "-")

        col4.metric("Akurasi Model", "78.3%")

        st.divider()

        chart_col1, chart_col2 = st.columns([1.3, 1])

        with chart_col1:
            st.subheader("Distribusi Sentimen")
            if sentiment_col is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(kind="bar", color=["#D85A30", "#1D9E75"], ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("Jumlah Ulasan")
                ax.set_title("Jumlah Ulasan per Sentimen")
                for i, v in enumerate(counts.values):
                    ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
                st.pyplot(fig)
            else:
                st.write("Kolom sentimen tidak ditemukan di dataset.")

        with chart_col2:
            st.subheader("Proporsi")
            if sentiment_col is not None:
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                counts.plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=["#D85A30", "#1D9E75"],
                    ylabel="",
                    ax=ax2,
                )
                st.pyplot(fig2)
            else:
                st.write("-")


# ===========================================================
# HALAMAN 2 — PREDIKSI SENTIMEN
# ===========================================================
else:
    st.title(" Prediksi Sentimen Ulasan")
    st.caption("TEGUH teks ulasan, sistem akan memprediksi sentimennya menggunakan Multinomial Naive Bayes")

    user_input = st.text_area(
        "Tulis ulasan di sini:",
        placeholder="Contoh: Aplikasinya sangat membantu transaksi sehari-hari, mudah dan cepat",
        height=140,
    )

    predict_btn = st.button("Prediksi Sentimen", type="primary", use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses..."):
                clean_text = preprocess_text(user_input)

            if not clean_text.strip():
                st.error(
                    "Teks tidak dapat diproses (kosong setelah preprocessing). "
                    "Coba masukkan teks yang lebih deskriptif."
                )
            else:
                X = vectorizer.transform([clean_text])
                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                prob_dict = dict(zip(model.classes_, proba))

                st.divider()

                if prediction == "Positif":
                    st.success(f"### Sentimen:  {prediction}")
                else:
                    st.error(f"### Sentimen:  {prediction}")

                pcol1, pcol2 = st.columns(2)
                pcol1.metric("Probabilitas Positif", f"{prob_dict.get('Positif', 0) * 100:.2f}%")
                pcol2.metric("Probabilitas Negatif", f"{prob_dict.get('Negatif', 0) * 100:.2f}%")

                with st.expander("Lihat detail preprocessing teks"):
                    st.write(f"**Teks asli:** {user_input}")
                    st.write(f"**Setelah preprocessing:** {clean_text}")

    st.divider()
    st.caption("Model: Multinomial Naive Bayes | TF-IDF (ngram 1-2, max_features=5000) | Akurasi uji: 78.25%")