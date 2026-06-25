import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
import gdown
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import zipfile

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="Digital Heritage Documentation",
    layout="wide"
)

# =========================
# تصميم بسيط
# =========================
st.markdown("""
<style>
.main { background-color: #f7f3ee; }
h1 { color: #6b3e26; }
h2, h3 { color: #2c5d7a; }
</style>
""", unsafe_allow_html=True)

# =========================
# المسارات
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# =========================
# تحميل النموذج من Google Drive
# =========================
file_id = "1wkiXLv04aJx7meYixVmtiD3DglomUh_E"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =========================
# تحميل Dataset (إذا غير موجود)
# =========================
DATASET_ZIP = os.path.join(BASE_DIR, "dataset.zip")

if not os.path.exists(DATASET_PATH):
    dataset_url = "https://drive.google.com/uc?id=1b7b5rMqHeEY4azw41v1x-30X73DxMyQw"
    gdown.download(dataset_url, DATASET_ZIP, quiet=False)

    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

# =========================
# قراءة الصور
# =========================
image_files = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# =========================
# عنوان التطبيق
# =========================
st.title("🏛️ Digital Heritage Documentation System")
st.header("📍 Bani Hammad Castle - UNESCO Heritage Site")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "Images", "Analysis", "AI Results", "Report"
])

# =========================
# TAB 1 - Images
# =========================
with tab1:
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(DATASET_PATH, img_name)
        st.image(Image.open(img_path), caption=f"Image {i+1}", use_container_width=True)

# =========================
# TAB 2 - Analysis
# =========================
with tab2:
    brightness_list = []

    for img_name in image_files:
        img = cv2.imread(os.path.join(DATASET_PATH, img_name))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_list.append(np.mean(gray))

    st.write(f"Total Images: {len(brightness_list)}")
    st.write(f"Average Brightness: {np.mean(brightness_list):.2f}")

    fig, ax = plt.subplots()
    ax.plot(brightness_list, marker='o')
    ax.set_title("Brightness Analysis")
    st.pyplot(fig)

# =========================
# TAB 3 - AI Prediction
# =========================
with tab3:
    classes = ["Front View", "Back View", "Left View", "Right View"]
    confidences = []

    for i, img_name in enumerate(image_files):

        img = cv2.imread(os.path.join(DATASET_PATH, img_name))
        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        index = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        confidences.append(confidence)

        st.success(f"Image {i+1} ➜ {classes[index]}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")
        st.divider()

    st.write(f"Average Confidence: {np.mean(confidences):.2f}%")

# =========================
# TAB 4 - REPORT
# =========================
with tab4:
    selected_images = []

    for i, img_name in enumerate(image_files):

        col1, col2 = st.columns([4, 1])

        with col1:
            st.image(os.path.join(DATASET_PATH, img_name), width=250)

        with col2:
            if st.checkbox(f"Select {i+1}", key=img_name):
                selected_images.append(img_name)

    if st.button("Generate Report"):

        if len(selected_images) == 0:
            st.warning("Please select at least one image")
            st.stop()

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Bani Hammad Castle Report", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)

        pdf.ln(10)

        for i, img_name in enumerate(selected_images, 1):

            path = os.path.join(DATASET_PATH, img_name)
            img = cv2.imread(path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Image {i}", ln=True)

            pdf.image(path, w=70)

            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"Brightness: {np.mean(gray):.2f}", ln=True)
            pdf.cell(0, 8, f"Contrast: {np.std(gray):.2f}", ln=True)

            pdf.ln(5)

        output_path = os.path.join(BASE_DIR, "report.pdf")
        pdf.output(output_path)

        st.success("Report generated successfully!")

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="heritage_report.pdf"
            )
