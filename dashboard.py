import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.pimport streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Custom Background & Style
# ==========================
st.set_page_config(page_title="Image Classification & Detection", page_icon="🧠", layout="centered")

# CSS untuk warna biru dashboard
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0078FF 0%, #004AAD 100%);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #003580;
}
[data-testid="stSidebar"] * {
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

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
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # ==========================
        # Hasil Prediksi (card putih di tengah)
        # ==========================
        st.markdown(
            f"""
            <div style="
                background-color:white;
                color:#004AAD;
                padding:25px;
                border-radius:15px;
                text-align:center;
                box-shadow:0px 4px 10px rgba(0,0,0,0.3);
                margin-top:25px;">
                <h2>🔹 Hasil Prediksi</h2>
                <h3>Kelas: {class_index}</h3>
                <p><b>Probabilitas:</b> {probability:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
reprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
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
st.set_page_config(page_title="Image Classification & Detection", page_icon="🧠", layout="centered")

st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        probability = np.max(prediction)

        # ==========================
        # Tampilan hasil prediksi (card biru elegan)
        # ==========================
        st.markdown(
            f"""
            <style>
            .result-card {{
                background: linear-gradient(135deg, #007BFF 0%, #0056D2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                margin-top: 25px;
            }}
            .result-title {{
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .result-text {{
                font-size: 20px;
                margin-top: 5px;
            }}
            </style>

            <div class="result-card">
                <div class="result-title">🔹 Hasil Prediksi</div>
                <div class="result-text">Kelas: <b>{class_index}</b></div>
                <div class="result-text">Probabilitas: <b>{probability:.2f}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
