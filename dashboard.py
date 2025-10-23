# ==========================
# DASHBOARD SMART IMAGE INSIGHT (FIX TANPA CV2)
# ==========================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ❗ Nonaktifkan import YOLO jika error, ganti alternatif
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception:
    yolo_available = False

# ==========================
# KONFIGURASI
# ==========================
st.set_page_config(page_title="🌸 Smart Image Insight Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #D0F0C0 0%, #B4E3B1 100%);
    color: #2E4031;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, p, label {
    color: #2E4031 !important;
}
.result-box {
    background-color: white;
    color: #2E4031;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Smart Image Insight Dashboard")
st.markdown("💚 *Deteksi objek dan lihat insight statistiknya secara otomatis!*")

# ==========================
# LOAD MODEL YOLO (jika tersedia)
# ==========================
if yolo_available:
    @st.cache_resource
    def load_yolo():
        return YOLO("yolov8n.pt")

    model = load_yolo()
else:
    st.warning("⚠️ YOLOv8 belum aktif (cv2 tidak tersedia). Menjalankan mode dummy untuk demo.")
    model = None

# ==========================
# MODE INPUT
# ==========================
mode = st.sidebar.radio("Pilih Sumber Gambar:", ["📁 Upload Gambar", "📸 Kamera Langsung"])

if mode == "📁 Upload Gambar":
    uploaded = st.file_uploader("Unggah Gambar di sini:", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    else:
        img = None
else:
    img = st.camera_input("Ambil Gambar dari Kamera")

# ==========================
# PROSES DETEKSI
# ==========================
if img:
    img_pil = Image.open(img).convert("RGB")
    st.image(img_pil, caption="📷 Gambar Asli", use_container_width=True)

    if yolo_available:
        with st.spinner("🔍 Mendeteksi objek..."):
            results = model(img_pil)
            boxes = results[0].boxes
            names = results[0].names
            labels = [names[int(cls)] for cls in boxes.cls]

            # Gunakan PIL untuk menampilkan hasil (tanpa cv2)
            result_img = Image.fromarray(results[0].plot()[:, :, ::-1])
    else:
        labels = ["person", "car", "dog"]  # dummy untuk demo
        result_img = img_pil

    # ==========================
    # TAMPILAN 3 KOLOM
    # ==========================
    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        st.image(result_img, caption="🟢 Hasil Deteksi YOLOv8", use_container_width=True)

    with col2:
        if labels:
            label_series = pd.Series(labels).value_counts()
            fig, ax = plt.subplots()
            label_series.plot(kind="barh", ax=ax)
            ax.set_xlabel("Jumlah Objek")
            ax.set_ylabel("Kategori")
            ax.set_title("📊 Distribusi Objek Terdeteksi")
            st.pyplot(fig)
        else:
            st.info("Tidak ada objek terdeteksi.")

    with col3:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("💡 Insight Otomatis")

        if labels:
            df_counts = pd.Series(labels).value_counts()
            most_common = df_counts.index[0]
            count = df_counts.iloc[0]
            st.write(f"🌟 Objek paling sering muncul: **{most_common}** sebanyak **{count} kali**.")
            st.write(f"🔎 Total objek terdeteksi: **{len(labels)}**.")
            if len(df_counts) > 1:
                st.write(f"Objek kedua terbanyak: **{df_counts.index[1]}**.")
            else:
                st.write("Hanya satu jenis objek yang terdeteksi.")
        else:
            st.write("Tidak ditemukan objek untuk dianalisis.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Silakan unggah atau ambil gambar terlebih dahulu.")
