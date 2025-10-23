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
# CEK LIBRARY SISTEM (UNTUK STREAMLIT)
# ==========================
try:
    import cv2
except:
    os.system("apt-get update -y && apt-get install -y libgl1 libglib2.0-0")

try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Library YOLO belum terinstal. Jalankan: pip install ultralytics")

# ==========================
# KONFIGURASI DASAR STREAMLIT
# ==========================
st.set_page_config(page_title="🧠 Image Detection & Classification", layout="centered")

# ==========================
# CSS WARNA BIRU
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
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None

    try:
        yolo_model = YOLO("model/best.pt")
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa memuat YOLOv8: {e}")

    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
    except Exception as e:
        st.warning(f"⚠️ Tidak bisa memuat model klasifikasi: {e}")

    return yolo_model, classifier

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

    if menu == "Deteksi Objek (YOLOv8)" and yolo_model:
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi YOLOv8", use_container_width=True)

    elif menu == "Klasifikasi Gambar" and classifier:
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
    st.info("📂 Silakan unggah gambar terlebih dahulu.")
