# ==========================================================
# 🌸 SmartVision Dashboard by Intan (UTS Big Data)
# ==========================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import random
import tempfile

# ⚙️ Coba import YOLOv8 (jika tidak ada, pakai mode demo)
try:
    from ultralytics import YOLO
    import cv2
    yolo_available = True
except:
    yolo_available = False

# 🎨 Setup halaman
st.set_page_config(page_title="🌿 SmartVision Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
    }
    .main {
        background-color: #f1f8e9;
        padding: 2rem;
        border-radius: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 SmartVision Dashboard")
st.markdown("_💚 Deteksi dan klasifikasi daun secara otomatis menggunakan YOLOv8 atau mode demo._")

# 🗂️ Sidebar menu
st.sidebar.header("🖼️ Pilih Mode Deteksi")
mode = st.sidebar.radio("Mode:", ["YOLOv8 (Deteksi Asli)", "Demo Klasifikasi Daun"])

menu = st.sidebar.radio("Sumber Gambar:", ["📁 Upload Gambar", "📸 Kamera Langsung"])

# Upload / Kamera
if menu == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar di sini:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None
else:
    image = st.camera_input("Ambil gambar:")

# ==========================================================
# 🧠 MODE YOLOv8
# ==========================================================
if mode == "YOLOv8 (Deteksi Asli)":
    st.subheader("🚀 Mode: YOLOv8 Detection")

    if not yolo_available:
        st.warning("⚠️ YOLOv8 belum aktif (cv2 tidak tersedia). Menjalankan mode demo untuk tampilan.")
    else:
        model_path = "best.pt"  # pastikan file YOLOv8 kamu ada di sini
        model = YOLO(model_path)

    if image is not None:
        # Simpan sementara untuk proses
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image.read() if hasattr(image, "read") else image.getvalue())
            img_path = temp_file.name

        if yolo_available:
            # Jalankan YOLOv8
            results = model(img_path)
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="🟢 Hasil Deteksi (YOLOv8)", use_column_width=True)

            # Ambil hasil deteksi
            data = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r.names[cls] if hasattr(r, "names") else f"Objek {cls}"
                    data.append({"Objek": label, "Probabilitas (%)": round(conf * 100, 2)})

            df = pd.DataFrame(data)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Objek"))
            else:
                st.info("Tidak ada objek terdeteksi.")
        else:
            st.info("Menampilkan mode demo karena YOLOv8 tidak aktif.")
            st.image(image, caption="Gambar Asli (Demo)", use_column_width=True)
    else:
        st.info("⬆️ Silakan unggah atau ambil gambar terlebih dahulu.")

# ==========================================================
# 🌿 MODE DEMO KLASIFIKASI DAUN
# ==========================================================
elif mode == "Demo Klasifikasi Daun":
    st.subheader("🌿 Mode: Demo Klasifikasi Daun")

    if image is not None:
        img_pil = Image.open(image)
        draw = ImageDraw.Draw(img_pil)

        # Simulasi deteksi daun
        daun_labels = ["Daun Sehat", "Daun Busuk", "Daun Terkena Jamur", "Daun Layu", "Daun Terinfeksi"]
        detected = random.sample(daun_labels, k=random.randint(1, 3))

        for obj in detected:
            x0, y0 = random.randint(50, 150), random.randint(50, 150)
            x1, y1 = x0 + random.randint(100, 250), y0 + random.randint(100, 250)
            draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
            draw.text((x0, y0 - 10), f"{obj}", fill="green")

        st.image(img_pil, caption="🟢 Hasil Deteksi (Simulasi Klasifikasi Daun)", use_column_width=True)

        # Data probabilitas acak
        df_demo = pd.DataFrame({
            "Objek": detected,
            "Probabilitas (%)": [round(random.uniform(80, 99), 2) for _ in detected]
        })

        st.dataframe(df_demo, use_container_width=True)
        st.bar_chart(df_demo.set_index("Objek"))
    else:
        st.info("⬆️ Silakan unggah gambar daun terlebih dahulu untuk melihat klasifikasi demo.")
