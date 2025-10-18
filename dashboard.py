import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# KONFIGURASI DASHBOARD (WARNA BIRU)
# ==========================
st.set_page_config(page_title="Image Detection & Classification", page_icon="🧠", layout="centered")

# Tambahkan CSS biru di sini ⬇️
page_style = """
<style>
/* ======== LATAR BELAKANG BIRU ======== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #007BFF 0%, #004AAD 100%);
    color: white;
}

/* ======== SIDEBAR ======== */
[data-testid="stSidebar"] {
    background-color: #003580;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* ======== PERBAIKAN INTERAKSI DROPDOWN & FILE UPLOADER ======== */
div[data-baseweb="select"],
.stFileUploader {
    z-index: 1000 !important;
    position: relative !important;
}
[data-testid="stSidebarNav"] {
    z-index: 1 !important;
}

/* ======== HEADER DAN TEKS ======== */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3, p, label {
    color: white !important;
}

/* ======== KOTAK HASIL ======== */
.result-card {
    background-color: white;
    color: #004AAD;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    margin-top: 25px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # ✅ Bagian f-string sudah ditutup dengan benar di bawah ini
        st.markdown(
            f"""
            <div class="result-card">
                <h2>🔹 Hasil Prediksi</h2>
                <h3>Kelas: {class_index}</h3>
                <p><b>Probabilitas:</b> {probability:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
