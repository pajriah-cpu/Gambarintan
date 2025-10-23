# ==========================
# DASHBOARD SMART IMAGE INSIGHT
# ==========================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ==========================
# KONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="🌸 Smart Image Insight Dashboard", layout="wide")

# ==========================
# TEMA STYLING FEMININ
# ==========================
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

# ==========================
# LOAD MODEL YOLOv8
# ==========================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# ==========================
# ANTARMUKA
# ==========================
st.title("🌸 Smart Image Insight Dashboard")
st.markdown("💚 *Deteksi objek dan lihat insight statistiknya secara otomatis!*")

mode = st.sidebar.radio("Pilih Sumber Gambar:", ["📁 Upload Gambar", "📸 Kamera Langsung"])

# Input Gambar
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

    with st.spinner("🔍 Mendeteksi objek..."):
        results = model(img_pil)
        result_image = results[0].plot()
        boxes = results[0].boxes
        names = results[0].names
        labels = [names[int(cls)] for cls in boxes.cls]

    # ==========================
    # TAMPILAN 3 KOLOM
    # ==========================
    col1, col2, col3 = st.columns([1.5, 1, 1])

    # Kolom 1: Hasil Deteksi
    with col1:
        st.image(result_image, caption="🟢 Hasil Deteksi YOLOv8", use_container_width=True)

    # Kolom 2: Grafik Jumlah Objek
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

    # Kolom 3: Insight Otomatis
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

    # ==========================
    # OPSIONAL: SIMPAN HASIL KE CSV
    # ==========================
    if labels:
        df = pd.DataFrame(labels, columns=["Object"])
        csv_path = "deteksi_log.csv"
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        st.success("📁 Hasil deteksi telah disimpan ke `deteksi_log.csv`.")
else:
    st.info("Silakan unggah atau ambil gambar terlebih dahulu.")
