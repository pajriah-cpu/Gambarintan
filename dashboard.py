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
# KONFIGURASI DASAR STREAMLIT
# ==========================
st.set_page_config(page_title="🌸 Image Detection & Classification", layout="centered")

# ==========================
# DESAIN WARNA LEMBUT (HIJAU PASTEL)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #B8E1C4 0%, #A0D8B3 100%);
    color: #2E4031;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #A8D5BA !important;
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
/* ======= Tombol Upload Cantik ======= */
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
# LOAD MODEL TANPA ERROR
# ==========================
@st.cache_resource
def load_models():
    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
        return classifier
    except:
        return None

classifier = load_models()

# ==========================
# ANTARMUKA UTAMA
# ==========================
st.title("🌸 Image Detection & Classification App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar yang Diupload", use_container_width=True)

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
else:
    st.info("📂 Silakan unggah gambar terlebih dahulu.")
