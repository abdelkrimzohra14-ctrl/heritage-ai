import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import pickle
from fpdf import FPDF   # ✔ هنا مهم جدًا

def predict_image(image):
    import numpy as np

    # تغيير الحجم حسب نموذجك (غالبًا 224x224)
    image = image.resize((224, 224))

    # تحويل الصورة إلى array
    img_array = np.array(image)

    # التأكد من 3 قنوات (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # تطبيع
    img_array = img_array.astype("float32") / 255.0

    # إضافة batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ

    classes = ["Front", "Back", "Left", "Right"]

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    return classes[index], confidence

# ==========================================
# إعداد الصفحة
# ==========================================
st.set_page_config(
    page_title="Digital Heritage Documentation",
    layout="wide"
)
import os


# =========================
# 🎨 تصميم الواجهة
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f3ee;
    }
    h1 {
        color: #6b3e26;
    }
    h2, h3 {
        color: #2c5d7a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 🏛️ الصور (شعار + قلعة)
# =========================
logo_path = "assets/University of M'sila.png"
castle_path = "assets/Bani Hammad Castle.png"

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image(logo_path, width=120)

with col2:
    st.image(castle_path, use_container_width=True)

with col3:
    st.write("")

# =========================
# 🏛️ العنوان
# =========================
st.markdown(
    """
    <h1 style='text-align:center; color:#6b3e26;'>
    🏛️ Digital Heritage Documentation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:#2c5d7a;'>
    AI-based Architectural Heritage Analysis
    </h4>
    """,
    unsafe_allow_html=True
)

# ==========================================
# المسارات
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
import gdown
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    model_url = "https://drive.google.com/uc?id=1jA80p388SNnuRiqfnQG7EvO3g0K21_mz"
    gdown.download(model_url, MODEL_PATH, quiet=False)

model = load_model("model.keras", compile=False)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# =========================
# تحميل Dataset (ZIP)
# =========================
DATASET_ZIP = os.path.join(BASE_DIR, "dataset.zip")

if not os.path.exists(DATASET_PATH):
    dataset_url = "https://drive.google.com/uc?id=1b7b5rMqHeEY4azw41v1x-30X73DxMyQw"
    gdown.download(dataset_url, DATASET_ZIP, quiet=False)

    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)



# ==========================================
# تحميل النموذج
# ==========================================
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Model Error : {e}")
    st.stop()

# ==========================================
# التأكد من وجود مجلد الصور
# ==========================================
if not os.path.exists(DATASET_PATH):
    st.error("Dataset folder not found.")
    st.stop()

# ==========================================
# قراءة الصور
# ==========================================
image_files = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# ==========================================
# العنوان
# ==========================================


st.header("📍 Site Information")
st.markdown("قلعة بني حماد - التراث العالمي")

# ==========================================
# معلومات الموقع
# ==========================================
st.header("📍 معلومات الموقع | Site Information")

st.markdown("""
### AR العربية

**الاسم:** قلعة بني حماد

**الموقع:** ولاية المسيلة – الجزائر

**تاريخ الإنشاء:** 1007م

**التصنيف:** التراث العالمي لليونسكو (1980)

تقع قلعة بني حماد شمال شرق ولاية المسيلة،
وكانت العاصمة الأولى للدولة الحمادية.

---

### ENG English

**Name:** Bani Hammad Castle

**Location:** M'Sila, Algeria

**Construction:** 1007 AD

**UNESCO:** World Heritage Site (1980)

It was the first capital of the Hammadid dynasty.
""")

# ==========================================
# معرض الصور
# ==========================================
st.header("AI-Based Documentation of Beni Hammad Heritage")

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Images",
    "📊 Analysis",
    "🤖 AI Results",
    "📄 Report"
])

with tab1:

    for i, image_name in enumerate(image_files):

        image_path = os.path.join(DATASET_PATH, image_name)
        image = Image.open(image_path)

        st.image(image, caption=f"Image {i+1}", use_container_width=True)

# ==========================================================
# Analysis
# ==========================================================
with tab2:

    brightness_list = []
    contrast_list = []

    for image_name in image_files:

        img = cv2.imread(os.path.join(DATASET_PATH, image_name))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness_list.append(np.mean(gray))
        contrast_list.append(np.std(gray))

    st.write(f"Total Images: {len(brightness_list)}")
    st.write(f"Average Brightness: {np.mean(brightness_list):.2f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(brightness_list, marker='o')

    ax.set_title("Heritage Image Brightness Analysis")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Brightness")

    fig.savefig("chart.png")
    st.pyplot(fig)

# ==========================================================
# AI Prediction
# ==========================================================
with tab3:

    classes = ["Front View", "Back View", "Left View", "Right View"]

    confidences = []

    for i, image_name in enumerate(image_files):

        img = cv2.imread(os.path.join(DATASET_PATH, image_name))
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
        st.write(f"📊 Confidence: {confidence:.2f}%")
        st.divider()

    st.subheader("📊 AI Summary")
    st.write(f"Average Confidence: {np.mean(confidences):.2f}%")

with tab4:
    st.subheader("📸 Select Images for Report (Maximum 10)")

    selected_images = []

    for i, image_name in enumerate(image_files):

        image_path = os.path.join(DATASET_PATH, image_name)

        col1, col2 = st.columns([4, 1])

        with col1:
            st.image(
            image_path,
            caption=f"Image {i+1}",
            width=250
        )

        with col2:
           if st.checkbox(
            f"Select {i+1}",
            key=image_name
        ):
            selected_images.append(image_name)

    st.write(f"Selected Images: {len(selected_images)}/10")
    st.subheader("📄 Generate Heritage Report")

    st.write(
        f"Selected Images: {len(selected_images)} / 10"
    )

    if st.button("📄 Generate Report"):

        if len(selected_images) == 0:
            st.warning("Please select at least one image.")
            st.stop()

        if len(selected_images) > 10:
            st.error("Maximum 10 images allowed.")
            st.stop()

        from fpdf import FPDF
        from datetime import datetime

        pdf = FPDF()
        pdf.add_page()

        # =====================
        # Title
        # =====================

        pdf.set_font("Arial", "B", 18)
        pdf.cell(
            0,
            10,
            "Bani Hammad Castle",
            ln=True,
            align="C"
        )

        pdf.set_font("Arial", "", 12)
        pdf.cell(
            0,
            10,
            "AI-Based Heritage Documentation",
            ln=True,
            align="C"
        )

        pdf.cell(
            0,
            10,
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            ln=True,
            align="C"
        )

        pdf.ln(10)

        # =====================
        # Selected Images
        # =====================

        for i, img_name in enumerate(selected_images, start=1):

            path = os.path.join(DATASET_PATH, img_name)

            img = cv2.imread(path)

            if img is None:
                continue

            if len(img.shape) == 2:
                gray = img
            else:
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Image {i}", ln=True)

            pdf.image(path, w=70)

            pdf.set_font("Arial", "", 11)
            pdf.cell(
                0,
                8,
                f"Brightness: {brightness:.2f}",
                ln=True
            )

            pdf.cell(
                0,
                8,
                f"Contrast: {contrast:.2f}",
                ln=True
            )

            pdf.ln(5)

        pdf_path = os.path.join(
            BASE_DIR,
            "BH_Heritage_Report.pdf"
        )

        pdf.output(pdf_path)

        st.success(
            "📄 Report generated successfully ✔"
        )

        with open(pdf_path, "rb") as file:

            st.download_button(
                "📥 Download Report",
                data=file,
                file_name="BH_Heritage_Report.pdf",
                mime="application/pdf"
            )
