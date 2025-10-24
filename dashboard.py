import streamlit as st
from PIL import Image
import numpy as np

# Cek import YOLO dan cv2
try:
    from ultralytics import YOLO
    import cv2
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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", use_column_width=True)

    if mode == "Klasifikasi":
        st.subheader("Mode: Klasifikasi")
        st.info("🔹 Gambar akan diklasifikasikan secara otomatis.")
        # Placeholder untuk model klasifikasi
        st.image(image, caption="Hasil Klasifikasi", use_column_width=True)
        st.write("✅ Kelas yang terdeteksi: **Daun Mangga**")  # contoh hasil klasifikasi

    else:
        st.subheader("Mode: Deteksi YOLOv8")
        st.info("🔹 Gambar akan dideteksi menggunakan YOLOv8.")

        try:
            # Load YOLOv8 model
            model = YOLO("yolov8n.pt")  # ganti path sesuai modelmu
            img_np = np.array(image)
            results = model.predict(img_np)
            results[0].save("temp_result.jpg")  # simpan hasil plot
            result_img = Image.open("temp_result.jpg")
            st.image(result_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

            # Tampilkan label hasil deteksi
            labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
            st.write(f"✅ Objek terdeteksi: {', '.join(labels)}")

        except Exception as e:
            st.error(f"Gagal menjalankan YOLOv8: {e}")
