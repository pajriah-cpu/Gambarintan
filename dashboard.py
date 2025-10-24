import streamlit as st
from PIL import Image
import numpy as np
import os

# Cek import YOLO dan cv2
try:
    from ultralytics import YOLO
    import cv2
    import tensorflow as tf
except ImportError as e:
    st.error(f"Module error: {e}")

# --- Style Background ---
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

# --- Sidebar ---
st.sidebar.header("Pilih Mode")
mode = st.sidebar.radio("Mode:", ["Deteksi YOLOv8", "Klasifikasi"])

st.sidebar.header("Sumber Gambar")
source = st.sidebar.radio("Sumber Gambar:", ["Upload Gambar", "Kamera Langsung"])

# --- Judul Dashboard ---
st.title("SmartVision Dashboard")
st.write("💚 Deteksi dan klasifikasi daun secara otomatis menggunakan YOLOv8 atau mode klasifikasi.")

# --- Penjelasan Tentang Mode ---
with st.expander("ℹ️ Penjelasan Tentang Mode"):
    st.markdown("""
    **🔍 Deteksi YOLOv8**
    - Digunakan untuk *mendeteksi objek* (misalnya daun, hama, atau penyakit) dalam gambar.  
    - YOLOv8 bekerja dengan menandai posisi objek menggunakan *bounding box* dan label kelas.  
    - Cocok digunakan untuk mengetahui **berapa banyak dan di mana objek berada** dalam satu gambar.

    **🧠 Klasifikasi**
    - Digunakan untuk *mengklasifikasikan satu gambar ke dalam kategori tertentu*.  
    - Misalnya untuk mengenali **jenis daun** atau **tingkat kesehatan tanaman**.  
    - Cocok ketika gambar hanya berisi satu objek utama dan kamu ingin tahu **jenis atau kondisinya**.
    """)

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Unggah Gambar di sini", type=["jpg", "jpeg", "png"])

# --- Path model ---
yolo_model_path = r"model/Intan Pajriah_Laporan 4.pt"
klasifikasi_model_path = r"model/intan_pajriah_laporan2.h5"

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    if mode == "Klasifikasi":
        st.subheader("Mode: Klasifikasi")
        st.info("🔹 Gambar akan diklasifikasikan menggunakan model kamu sendiri.")
        try:
            model_klasifikasi = tf.keras.models.load_model(klasifikasi_model_path)
            img_resized = image.resize((224, 224))  # ubah sesuai input model kamu
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            pred = model_klasifikasi.predict(img_array)
            kelas = np.argmax(pred, axis=1)[0]

            # ganti label sesuai urutan kelas model kamu
            label_kelas = ["Daun Mangga", "Daun Jambu", "Daun Jeruk"]
            st.image(image, caption="Hasil Klasifikasi", use_column_width=True)
            st.success(f"✅ Kelas yang terdeteksi: **{label_kelas[kelas]}**")

        except Exception as e:
            st.error(f"Gagal menjalankan model klasifikasi: {e}")

    else:
        st.subheader("Mode: Deteksi YOLOv8")
        st.info("🔹 Gambar akan dideteksi menggunakan model YOLOv8 kamu sendiri.")
        try:
            model_yolo = YOLO(yolo_model_path)
            img_np = np.array(image)
            results = model_yolo.predict(img_np)
            results[0].save("temp_result.jpg")
            result_img = Image.open("temp_result.jpg")
            st.image(result_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

            labels = [model_yolo.names[int(cls)] for cls in results[0].boxes.cls]
            st.success(f"✅ Objek terdeteksi: {', '.join(labels)}")

        except Exception as e:
            st.error(f"Gagal menjalankan YOLOv8: {e}")
