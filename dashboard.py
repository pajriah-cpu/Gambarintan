import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# CONFIGURASI DASHBOARD
# ==========================
st.set_page_config(page_title="Image Detection & Classification", page_icon="🧠", layout="centered")

# Warna biru background untuk seluruh dashboard
page_style = """
<style>
/* Background utama */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #007BFF 0%, #004AAD 100%);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #003580;
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Teks umum */
h1, h2, h3, p, label {
    color: white !important;
}

/* Kotak hasil prediksi */
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
        # ==========================
        # DETEKSI OBJEK DENGAN YOLO
        # ==========================
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan bounding box)
        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # ==========================
        # KLASIFIKASI GAMBAR
        # ==========================
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # ==========================
        # TAMPILKAN HASIL DI KOTAK PUTIH
        # ==========================
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
