import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import random
import pandas as pd

# 🎨 Setup halaman
st.set_page_config(page_title="Smart Image Insight Dashboard", layout="wide")

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

st.title("🌸 Smart Image Insight Dashboard")
st.markdown("_💚 Deteksi objek dan lihat insight statistiknya secara otomatis (Mode Demo)._")

# 🗂️ Sidebar
st.sidebar.header("Pilih Sumber Gambar:")
menu = st.sidebar.radio("Opsi Input", ["📁 Upload Gambar", "📸 Kamera Langsung"])

# 📤 Upload atau ambil gambar
if menu == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar di sini:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None
else:
    image = st.camera_input("Ambil gambar dengan kamera:")

# 🧠 Simulasi “deteksi objek”
if image is not None:
    image_pil = Image.open(image)
    draw = ImageDraw.Draw(image_pil)

    # Simulasikan deteksi objek (dummy bounding box)
    classes = ["Orang", "Mobil", "Kucing", "Anjing", "Botol", "Kursi", "Laptop"]
    detected = random.sample(classes, k=random.randint(2, 5))

    for obj in detected:
        x0, y0 = random.randint(0, 100), random.randint(0, 100)
        x1, y1 = x0 + random.randint(50, 200), y0 + random.randint(50, 200)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0 - 10), obj, fill="red")

    st.image(image_pil, caption="🟢 Hasil Deteksi Objek (Simulasi)", use_column_width=True)

    # 📊 Statistik hasil deteksi
    st.subheader("📈 Insight Deteksi Gambar")
    data = pd.DataFrame({
        "Objek": detected,
        "Probabilitas (%)": [round(random.uniform(70, 99), 2) for _ in detected]
    })

    st.dataframe(data, use_container_width=True)

    # Tambahkan grafik batang
    st.bar_chart(data.set_index("Objek"))
else:
    st.info("⬆️ Silakan unggah atau ambil gambar terlebih dahulu untuk melihat hasil deteksi.")
