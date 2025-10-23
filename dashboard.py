import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import random
import pandas as pd

# 🎨 Setup halaman
st.set_page_config(page_title="🌿 Smart Leaf Classifier", layout="wide")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
        color: #2e4031;
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #b4e3b1 0%, #d7ead3 100%) !important;
        color: #2e4031 !important;
    }
    h1, h2, h3, p, label {
        color: #2e4031 !important;
    }
    .result-card {
        background-color: white;
        color: #2e4031;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-top: 25px;
    }
    [data-testid="stFileUploader"] div[role="button"] {
        background-color: #ffffff !important;
        color: #3c6e47 !important;
        font-weight: 600;
        border-radius: 10px;
        border: 2px solid #3c6e47;
        transition: 0.3s;
    }
    [data-testid="stFileUploader"] div[role="button"]:hover {
        background-color: #3c6e47 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# 🏷️ Judul Utama
# =============================
st.title("🌿 Smart Leaf Classifier & Insight Dashboard")
st.markdown("_💚 Klasifikasikan jenis daun dan lihat insight visualnya dengan gaya lembut dan elegan._")

# =============================
# 🗂️ Sidebar Menu
# =============================
st.sidebar.header("🌸 Pilih Mode:")
mode = st.sidebar.radio("Mode Operasi:", ["Klasifikasi Daun", "Deteksi Area Daun (Simulasi)"])

uploaded_file = st.file_uploader("📸 Unggah Gambar Daun", type=["jpg", "jpeg", "png"])

# =============================
# 📷 Jika Gambar Diupload
# =============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # perbaiki kontras biar daun terlihat lebih jelas
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    st.image(image, caption="🌿 Gambar Daun yang Diupload", use_container_width=True)

    # =======================================
    # 🌿 MODE 1: KLASIFIKASI DAUN
    # =======================================
    if mode == "Klasifikasi Daun":
        st.subheader("🧠 Hasil Klasifikasi (Simulasi AI)")

        # contoh label daun
        leaf_types = ["Daun Sehat", "Daun Layu", "Daun Sakit (Jamur)", "Daun Kering"]
        predicted_label = random.choice(leaf_types)
        confidence = round(random.uniform(80, 99.5), 2)

        st.markdown(f"""
            <div class="result-card">
                <h2>🌱 Prediksi: {predicted_label}</h2>
                <p><b>Tingkat Kepercayaan:</b> {confidence}%</p>
            </div>
        """, unsafe_allow_html=True)

        # buat tabel probabilitas
        df = pd.DataFrame({
            "Kategori Daun": leaf_types,
            "Probabilitas (%)": [round(random.uniform(50, 100), 2) for _ in leaf_types]
        }).sort_values("Probabilitas (%)", ascending=False)

        st.subheader("📊 Probabilitas Klasifikasi Daun")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Kategori Daun"))

    # =======================================
    # 🌿 MODE 2: DETEKSI AREA DAUN
    # =======================================
    elif mode == "Deteksi Area Daun (Simulasi)":
        st.subheader("🌿 Simulasi Area Daun Terdeteksi")

        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size

        for _ in range(random.randint(1, 4)):
            x0 = random.randint(0, int(img_w * 0.6))
            y0 = random.randint(0, int(img_h * 0.6))
            x1 = x0 + random.randint(int(img_w * 0.2), int(img_w * 0.4))
            y1 = y0 + random.randint(int(img_h * 0.2), int(img_h * 0.4))
            color = random.choice(["#32CD32", "#66BB6A", "#81C784", "#A5D6A7"])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            draw.text((x0 + 10, y0 - 25), "Daun", fill=color)

        st.image(image, caption="🟢 Area Daun (Deteksi Simulasi)", use_container_width=True)

else:
    st.info("⬆️ Silakan unggah gambar daun terlebih dahulu untuk melihat hasil klasifikasi atau deteksi.")
