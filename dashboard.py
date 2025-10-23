# ==========================
# IMPORT LIBRARY
# ==========================
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from PIL import Image
import cv2  # pastikan opencv-python-headless terinstal

# ==========================
# KONFIGURASI DASHBOARD
# ==========================
st.set_page_config(page_title="Image Detection & Classification", page_icon="🧠", layout="centered")

# ==========================
# CSS WARNA BIRU LEMBUT
# ==========================
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #66B2FF 0%, #0055CC 100%);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #468CE8 !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1, h2, h3, p, label {
    color: white !important;
}
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
# LOAD MODEL (DICACHE)
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # Model YOLO
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        yolo_model = None

    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model Klasifikasi
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI DASHBOARD
# ==========================
st.title("🧠 Image Detection & Classification App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is not None:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True_
