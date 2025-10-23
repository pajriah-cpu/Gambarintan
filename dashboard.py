# ==========================
# IMPORT LIBRARY
# ==========================
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ==========================
# CEK LIBRARY OPSIONAL (YOLO & OPENCV)
# ==========================
yolo_model = None
try:
    from ultralytics import YOLO
    import cv2
except:
    st.warning("⚠️ Mode deteksi YOLO tidak aktif karena library 'ultralytics' belum terinstal. "
               "Anda tetap bisa menggunakan mode klasifikasi gambar.")
    YOLO = None

# ==========================
# KONFIGURASI DASAR STREAMLIT
# ==========================
st.set_page_config(page_title="🧠 Image Detection & Classification", layout="centered")

# ==========================
# CSS DESAIN + WARNA TOMBOL
# ==========================
st.markdown("""
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
/* ======= ubah warna tombol upload ======= */
[data-testid="stFileUploader"] div[role="button"] {
    background-color: #ffffff !important;
    color: #004AAD !important;
    font-weight: 600;
    border-radius: 10px;
    border: 2px solid #004AAD;
    transition: 0.3s;
}
[data-testid="stFileUploader"] div[role="button"]:hover {
    background-color: #004AAD !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL (AMAN TANPA ERROR)
# ==========================
@st.cache_resource
def load_models():
    yolo = None
    classifier = None

    # Model YOLO (kalau tersedia)
    if YOLO is not None:
        try:
            yolo = YOLO("model/best.pt")
        except Exception as e:
            st.info(f"ℹ️ YOLO belum tersedia: {e}")

    # Model klasifikasi
    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
    except Exception:
        st.info("ℹ️ Model klasifikasi belum ditemukan, tetapi aplikasi tetap dapat dibuka.")

    return yolo, classifier


yolo_model, classifier = load_models()

# ==========================
# ANTARMUKA UTAMA
# ==========================
st.title("🧠 Image Detection & Classification App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLOv8)", "Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLOv8)":
        if yolo_model:
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="📦 Hasil Deteksi YOLOv8", use_container_width=True)
        else:
            st.warning("⚠️ Model YOLO belum tersedia. Jalankan mode klasifikasi saja.")

    elif menu == "Klasifikasi Gambar":
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
                    <h2>🔹 Hasil Klasifikasi</h2>
                    <h3>Kelas: {class_index}</h3>
                    <p><b>Probabilitas:</b> {probability:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model klasifikasi belum tersedia.")
else:
    st.info("📂 Silakan unggah gambar terlebih dahulu.")
