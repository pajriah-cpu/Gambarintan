import streamlit as st
from PIL import Image
import cv2
import numpy as np

# --- Style Background ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E0F7FA;  /* Ganti warna sesuai selera */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
st.sidebar.image("2ead9cef-2448-47d3-9aec-09654ac231e1.png", use_column_width=True)
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
        
        # --- Simulasi output klasifikasi ---
        st.image(image, caption="Hasil Klasifikasi", use_column_width=True)
        st.write("✅ Kelas yang terdeteksi: **Daun Mangga**")  # contoh, sesuaikan dengan model

    else:  # Mode Deteksi YOLOv8
        st.subheader("Mode: Deteksi YOLOv8")
        st.info("🔹 Gambar akan dideteksi menggunakan YOLOv8.")

        # --- Simulasi deteksi YOLOv8 dengan bounding box ---
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Contoh koordinat bounding box [x1, y1, x2, y2]
        boxes = [[50, 50, 200, 200], [220, 80, 370, 230]]
        labels = ["Daun Mangga", "Daun Jambu"]
        colors = [(0, 255, 0), (0, 0, 255)]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), colors[i], 2)
            cv2.putText(image_cv, labels[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2)

        image_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(image_result, caption="Hasil Deteksi YOLOv8", use_column_width=True)
        st.write(f"✅ Objek terdeteksi: {', '.join(labels)}")
