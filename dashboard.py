# ======================================================
# 🌸 DASHBOARD AKHIR UTS BIG DATA - INTAN PAJRIAH
# ======================================================
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ======================================================
# 🎨 KONFIGURASI DASAR & WARNA FEMININ
# ======================================================
st.set_page_config(page_title="🌿 Smart Image Insight Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #D0F0C0 0%, #B4E3B1 100%);
        color: #2E4031;
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #B4E3B1 0%, #D7EAD3 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #2E4031 !important;
    }
    h1, h2, h3, p, label {
        color: #2E4031 !important;
    }
    .result-card {
        background-color: white;
        color: #2E4031;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-top: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Smart Image Insight Dashboard")
st.markdown("_💚 Deteksi dan klasifikasi gambar daun dengan tampilan lembut dan interaktif._")

# ======================================================
# 🧩 CEK YOLOv8 TERSEDIA ATAU TIDAK
# ======================================================
try:
    from ultralytics import YOLO
    yolo_available = True
except:
    yolo_available = False

# ======================================================
# 🧠 LOAD MODEL KLASIFIKASI
# ======================================================
@st.cache_resource
def load_classifier():
    try:
        model = tf.keras.models.load_model("model/classifier_model.h5")
        return model
    except:
        return None

classifier = load_classifier()

# ======================================================
# 🧭 PILIH MODE
# ======================================================
st.sidebar.header("⚙️ Pilih Mode Deteksi:")
mode = st.sidebar.selectbox("Pilih Fitur:", [
    "📊 Klasifikasi Daun",
    "🔍 Deteksi Objek (YOLOv8 Asli)",
    "🎨 Deteksi Objek (Simulasi YOLOv8)"
])

uploaded_file = st.file_uploader("📸 Unggah Gambar Daun atau Objek", type=["jpg", "jpeg", "png"])

# ======================================================
# 🚀 PROSES GAMBAR
# ======================================================
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="📷 Gambar yang Diupload", use_container_width=True)

    # ======================================================
    # 📊 MODE 1: KLASIFIKASI DAUN
    # ======================================================
    if mode == "📊 Klasifikasi Daun":
        if classifier:
            st.info("🤖 Sedang melakukan klasifikasi...")
            img_resized = image_pil.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            try:
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)
            except Exception as e:
                st.error(f"Terjadi kesalahan prediksi: {e}")
                st.stop()

            st.markdown(f"""
                <div class="result-card">
                    <h2>🌿 Hasil Klasifikasi Daun</h2>
                    <h3>Kelas Prediksi: <b>{class_index}</b></h3>
                    <p><b>Akurasi:</b> {probability:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Model klasifikasi belum ditemukan! Pastikan file 'classifier_model.h5' ada di folder 'model/'.")

    # ======================================================
    # 🔍 MODE 2: YOLOv8 ASLI
    # ======================================================
    elif mode == "🔍 Deteksi Objek (YOLOv8 Asli)":
        if yolo_available:
            try:
                st.info("🚀 YOLOv8 aktif! Sedang memproses deteksi objek...")
                model = YOLO("yolov8n.pt")
                results = model(image_pil)
                result_image = results[0].plot()
                st.image(result_image, caption="🟢 Hasil Deteksi YOLOv8", use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ YOLOv8 tidak dapat dijalankan ({e}). Menampilkan mode simulasi.")
        else:
            st.warning("⚠️ YOLOv8 belum terinstal di sistem. Menampilkan mode simulasi otomatis.")

            # fallback ke simulasi
            image_copy = image_pil.copy()
            draw = ImageDraw.Draw(image_copy)
            classes = ["Daun Hijau", "Daun Kering", "Bintik", "Batang", "Latar"]
            detected = random.sample(classes, k=random.randint(2, 4))

            for obj in detected:
                x0, y0 = random.randint(10, 100), random.randint(10, 100)
                x1, y1 = x0 + random.randint(100, 250), y0 + random.randint(80, 200)
                prob = round(random.uniform(70, 99), 2)
                draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
                draw.text((x0, y0 - 10), f"{obj} ({prob}%)", fill="green")

            st.image(image_copy, caption="🟢 Hasil Deteksi (Simulasi YOLOv8)", use_container_width=True)

    # ======================================================
    # 🎨 MODE 3: SIMULASI YOLOv8
    # ======================================================
    elif mode == "🎨 Deteksi Objek (Simulasi YOLOv8)":
        image_copy = image_pil.copy()
        draw = ImageDraw.Draw(image_copy)
        classes = ["Daun Sehat", "Daun Busuk", "Bercak", "Latar", "Batang"]
        detected = random.sample(classes, k=random.randint(2, 4))

        data = []
        for obj in detected:
            x0, y0 = random.randint(20, 120), random.randint(20, 120)
            x1, y1 = x0 + random.randint(100, 200), y0 + random.randint(80, 180)
            prob = round(random.uniform(70, 99), 2)
            draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
            draw.text((x0, y0 - 10), f"{obj} ({prob}%)", fill="green")
            data.append({"Objek": obj, "Probabilitas (%)": prob})

        st.image(image_copy, caption="🟢 Hasil Deteksi (Mode Simulasi)", use_container_width=True)
        df = pd.DataFrame(data)
        st.subheader("📊 Statistik Deteksi Objek")
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Objek"))

else:
    st.info("📂 Silakan unggah gambar terlebih dahulu untuk mulai deteksi.")
