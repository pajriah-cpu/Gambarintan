import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd

# Coba aktifkan YOLO
try:
    from ultralytics import YOLO
    import cv2
    yolo_active = True
    model = YOLO("yolov8n.pt")  # model kecil agar ringan
except Exception as e:
    yolo_active = False

# 🎨 Setup halaman
st.set_page_config(page_title="🌿 Deteksi Daun Pintar", layout="wide")

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

st.title("🌿 Smart Leaf Detection Dashboard")
st.markdown("_💚 Deteksi kondisi atau jenis daun secara otomatis menggunakan YOLOv8 (atau mode demo jika tidak aktif)._")

# 🗂️ Sidebar
st.sidebar.header("Pilih Gambar:")
menu = st.sidebar.radio("Sumber Gambar", ["📁 Upload", "📸 Kamera"])

if menu == "📁 Upload":
    uploaded_file = st.file_uploader("Unggah Gambar Daun:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None
else:
    image = st.camera_input("Ambil Gambar Daun")

# 🧠 Proses deteksi
if image is not None:
    image_pil = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(image_pil)

    if yolo_active:
        st.success("✅ YOLOv8 aktif! Menjalankan deteksi sebenarnya...")
        # Konversi gambar untuk YOLO
        img_array = np.array(image_pil)
        results = model.predict(source=img_array, conf=0.5, verbose=False)

        detected_objects = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = results[0].names[int(box.cls[0])]
            detected_objects.append((label, conf))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{label} ({conf*100:.1f}%)", fill="red")

    else:
        st.warning("⚠️ YOLOv8 belum aktif (cv2 tidak tersedia). Menjalankan mode demo untuk tampilan.")
        classes = ["Daun Sehat", "Daun Sakit", "Daun Layu", "Daun Kering"]
        detected_objects = random.sample(classes, k=random.randint(1, 3))

        for obj in detected_objects:
            x0, y0 = random.randint(50, 150), random.randint(50, 150)
            x1, y1 = x0 + random.randint(100, 200), y0 + random.randint(80, 150)
            conf = round(random.uniform(0.75, 0.98), 2)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0 - 15), f"{obj} ({conf*100:.1f}%)", fill="red")

        # ubah ke format dataframe untuk demo
        detected_objects = [(obj, random.uniform(0.75, 0.98)) for obj in detected_objects]

    # 🖼️ Tampilkan hasil deteksi
    st.image(image_pil, caption="🟢 Hasil Deteksi Daun", use_column_width=True)

    # 📊 Tabel hasil deteksi
    st.subheader("📈 Hasil Deteksi dan Confidence")
    df = pd.DataFrame({
        "Objek / Klasifikasi": [obj[0] for obj in detected_objects],
        "Confidence (%)": [round(obj[1]*100, 2) for obj in detected_objects]
    })
    st.dataframe(df, use_container_width=True)

    # 📊 Grafik batang
    st.bar_chart(df.set_index("Objek / Klasifikasi"))
else:
    st.info("⬆️ Silakan unggah atau ambil gambar daun terlebih dahulu untuk mendeteksi.")
