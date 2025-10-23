import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# ----------------- Style Background -----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E0F7FA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Sidebar -----------------
st.sidebar.header("Pilih Mode")
mode = st.sidebar.radio("Mode:", ["Deteksi YOLOv8", "Klasifikasi"])

st.sidebar.header("Sumber Gambar")
source = st.sidebar.radio("Sumber Gambar:", ["Upload Gambar"])  # Kamera bisa ditambahkan nanti

# ----------------- Judul -----------------
st.title("SmartVision Dashboard")
st.write("💚 Deteksi dan klasifikasi daun otomatis menggunakan YOLOv8 atau model klasifikasi.")

# ----------------- Upload Gambar -----------------
uploaded_file = st.file_uploader("Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    if mode == "Klasifikasi":
        st.subheader("Mode: Klasifikasi")
        st.info("🔹 Gambar akan diklasifikasikan secara otomatis.")

        # Load model klasifikasi (pastikan file ada di folder model)
        model_cls = tf.keras.models.load_model("model/klasifikasi_model.h5")
        img_array = np.array(image.resize((224, 224)))/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model_cls.predict(img_array)
        class_idx = np.argmax(prediction)
        st.image(image, caption="Hasil Klasifikasi", use_column_width=True)
        st.write(f"✅ Kelas yang terdeteksi: **Kelas {class_idx}**")  # Ganti sesuai label nyata

    else:
        st.subheader("Mode: Deteksi YOLOv8")
        st.info("🔹 Gambar akan dideteksi menggunakan YOLOv8.")

        # Load YOLOv8 model (pastikan file ada di folder model)
        model_yolo = YOLO("model/yolov8n.pt")

        img_np = np.array(image)
        results = model_yolo.predict(img_np)

        # Tampilkan hasil deteksi
        result_img = results[0].plot()  # Plot dengan bounding box
        st.image(result_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

        # Tampilkan label yang terdeteksi
        labels = [model_yolo.names[int(cls)] for cls in results[0].boxes.cls]
        if labels:
            st.write(f"✅ Objek terdeteksi: {', '.join(labels)}")
        else:
            st.write("❌ Tidak ada objek terdeteksi")
