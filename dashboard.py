import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

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

# --- Upload Gambar ---
uploaded_file = st.file_uploader("Unggah Gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    if mode == "Klasifikasi":
        st.subheader("Mode: Klasifikasi")
        st.info("🔹 Gambar akan diklasifikasikan secara otomatis.")
        # Panggil model klasifikasi kamu di sini
        st.image(image, caption="Hasil Klasifikasi", use_column_width=True)
        st.write("✅ Kelas yang terdeteksi: **Daun Mangga**")  # contoh

    else:
        st.subheader("Mode: Deteksi YOLOv8")
        st.info("🔹 Gambar akan dideteksi menggunakan YOLOv8.")

        # --- Load model YOLOv8 (misal model custom kamu) ---
        model = YOLO("yolov8n.pt")  # ganti path ke modelmu, misal "runs/train/exp/weights/best.pt"

        # Convert PIL image ke numpy array
        img_np = np.array(image)

        # Jalankan deteksi
        results = model.predict(img_np)

        # Ambil hasil gambar dengan bounding box
        result_img = results[0].plot()  # ini langsung menampilkan box + label
        st.image(result_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

        # Tampilkan label yang terdeteksi
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
        st.write(f"✅ Objek terdeteksi: {', '.join(labels)}")
