# ==========================
# IMPORT LIBRARY
# ==========================
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Coba load YOLOv8
try:
    from ultralytics import YOLO
    yolo_available = True
except:
    yolo_available = False

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🌸 Image Detection & Classification", layout="centered")

# ==========================
# TEMA WARNA FEMININ HIJAU PASTEL
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #D0F0C0 0%, #B4E3B1 100%);
    color: #2E4031;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #B4E3B1 !important;
}
[data-testid="stSidebar"] * {
    color: #2E4031 !important;
}
h1, h2, h3, p, label {
    color: #2E4031 !important;
}
.result-card {
    background-color: white;
    color: #2E4031;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-top: 25px;
}
/* Tombol Upload Cantik */
[data-testid="stFileUploader"] div[role="button"] {
    background-color: #ffffff !important;
    color: #3C6E47 !important;
    font-weight: 600;
    border-radius: 10px;
    border: 2px solid #3C6E47;
    transition: 0.3s;
}
[data-testid="stFileUploader"] div[role="button"]:hover {
    background-color: #3C6E47 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
    except:
        classifier = None

    if yolo_available:
        try:
            yolo_model = YOLO("yolov8n.pt")
        except:
            yolo_model = None
    else:
        yolo_model = None

    return classifier, yolo_model

classifier, yolo_model = load_models()

# ==========================
# ANTARMUKA UTAMA
# ==========================
st.title("🌸 Image Detection & Classification App")

mode = st.sidebar.selectbox("Pilih Mode:", ["Klasifikasi Gambar", "Deteksi Objek (YOLOv8)"])
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

    if mode == "Klasifikasi Gambar":
        if classifier:
            with st.spinner("🤖 Sedang melakukan klasifikasi..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)

            st.markdown(f"""
                <div class="result-card">
                    <h2>💚 Hasil Klasifikasi</h2>
                    <h3>Kelas: {class_index}</h3>
                    <p><b>Probabilitas:</b> {probability:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model klasifikasi belum ditemukan. Silakan unggah model terlebih dahulu.")

    elif mode == "Deteksi Objek (YOLOv8)":
        if yolo_available and yolo_model:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_image = results[0].plot()  # hasil gambar deteksi
                st.image(result_image, caption="🟢 Hasil Deteksi Y_
