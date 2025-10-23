# ===========================================================
# SMART IMAGE INSIGHT DASHBOARD (SOFT GREEN VERSION)
# ===========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Coba load YOLO
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception:
    yolo_available = False

# =======================
# SETTING DASBOR
# =======================
st.set_page_config(
    page_title="🌸 Smart Image Insight Dashboard",
    layout="wide",
    page_icon="🌷"
)

# CSS KUSTOM (agar sidebar dan halaman utama senada)
st.markdown("""
<style>
/* Warna latar belakang halaman utama */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #CDECCD 0%, #B2E2B0 100%);
    color: #2E4031;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #BFE8C4 0%, #A9DBA7 100%);
    color: #2E4031;
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
}

/* Heading dan label */
h1, h2, h3, label {
    color: #2E4031 !important;
}

/* Box hasil */
.result-box {
    background-color: white;
    border-radius: 20px;
    padding: 20px;
    color: #2E4031;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
    text-align: center;
}

/* Peringatan lembut */
[data-testid="stNotification"] {
    background-color: #F0FFE8 !important;
    border-left: 4px solid #A0CFA0 !important;
}
</style>
""", unsafe_allow_html=True)

# =======================
# JUDUL UTAMA
# =======================
st.title("🌸 Smart Image Insight Dashboard")
st.markdown("💚 *Deteksi objek dan lihat insight statistiknya secara otomatis!*")

# =======================
# INFO YOLO
# =======================
if yolo_available:
    st.success("✅ YOLOv8 aktif. Anda dapat melakukan deteksi objek dengan model asli.")
else:
    st.warning("⚠️ YOLOv8 belum aktif (cv2 tidak tersedia). Menjalankan mode demo untuk tampilan.")

# =======================
# MODE INPUT GAMBAR
# =======================
mode = st.sidebar.radio(
    "📂 Pilih Sumber Gambar:",
    ["🖼️ Upload Gambar", "📸 Kamera Langsung"]
)

if mode == "🖼️ Upload Gambar":
    uploaded = st.file_uploader("Unggah Gambar di sini:", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    else:
        img = None
else:
    img = st.camera_input("Ambil Gambar dari Kamera")

# =======================
# PROSES DETEKSI
# =======================
if img:
    st.image(img, caption="📷 Gambar Asli", use_container_width=True)

    if yolo_available:
        with st.spinner("🔍 Mendeteksi objek..."):
            model = YOLO("yolov8n.pt")
            results = model(img)
            boxes = results[0].boxes
            names = results[0].names
            labels = [names[int(cls)] for cls in boxes.cls]
            result_img = Image.fromarray(results[0].plot()[:, :, ::-1])
    else:
        labels = ["person", "cat", "car"]  # dummy
        result_img = img

    # =======================
    # TAMPILAN HASIL 3 KOLOM
    # =======================
    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        st.image(result_img, caption="🟢 Hasil Deteksi", use_container_width=True)

    with col2:
        if labels:
            label_series = pd.Series(labels).value_counts()
            fig, ax = plt.subplots()
            label_series.plot(kind="barh", ax=ax, color="#7FB77E")
            ax.set_xlabel("Jumlah")
            ax.set_ylabel("Kategori")
            ax.set_title("📊 Distribusi Objek")
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
            st.write(f"🌟 Objek terbanyak: **{most_common}** sebanyak **{count}** kali.")
            st.write(f"🔎 Total objek: **{len(labels)}**")
            if len(df_counts) > 1:
                st.write(f"🥈 Objek kedua: **{df_counts.index[1]}**")
        else:
            st.write("Belum ada objek terdeteksi.")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("🌷 Silakan unggah atau ambil gambar terlebih dahulu.")
