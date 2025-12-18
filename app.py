import streamlit as st
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64
from streamlit_paste_button import paste_image_button

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "/home/user171125/Documents/model testing/best copy 4.pt"
CONF_THRESHOLD = 0.15          # tuned
INFER_IMG_SIZE = 960        # higher = better small-object detection

st.set_page_config(page_title="Food Detection", layout="centered")
st.title("🍽️ Fruits & Veg DETECTION App")

# ======================================================
# LOAD MODEL (CACHED)
# ======================================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ======================================================
# IMAGE PREPROCESSING (SAFE FOR YOLO)
# ======================================================
def resize_image(img, max_size=1280):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_image(pil_img):
    img = np.array(pil_img)
    img = resize_image(img)
    img = enhance_contrast(img)
    img = denoise(img)
    # img = sharpen(img)   # Uncomment ONLY if images are blurry
    return Image.fromarray(img)

# ======================================================
# STATE
# ======================================================
if "image" not in st.session_state:
    st.session_state.image = None

if "run_detect" not in st.session_state:
    st.session_state.run_detect = False

# ======================================================
# INPUTS
# ======================================================
st.markdown("### 📥 Input Image")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

image_url = st.text_input(
    "OR paste image URL (http / https / data:image)",
    placeholder="https://example.com/food.jpg OR data:image/jpeg;base64,..."
)

st.markdown("OR copy & paste image below 👇")
paste_result = paste_image_button("📋 Paste Image")

# ======================================================
# LOAD IMAGE
# ======================================================
if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

elif image_url:
    try:
        if image_url.startswith("data:image"):
            header, encoded = image_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            st.session_state.image = Image.open(BytesIO(img_bytes)).convert("RGB")
        else:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            st.session_state.image = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        st.error("❌ Could not load image from URL")

elif paste_result.image_data is not None:
    if isinstance(paste_result.image_data, bytes):
        st.session_state.image = Image.open(BytesIO(paste_result.image_data)).convert("RGB")
    else:
        st.session_state.image = paste_result.image_data.convert("RGB")

# ======================================================
# BUTTON
# ======================================================
if st.button("🚀 Start Detecting"):
    if st.session_state.image is None:
        st.warning("⚠️ Please upload / paste an image first.")
    else:
        st.session_state.run_detect = True

# ======================================================
# YOLO DETECTION
# ======================================================
if st.session_state.run_detect:

    # 🔥 PREPROCESS IMAGE
    original_image = st.session_state.image
    processed_image = preprocess_image(original_image)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        processed_image.save(tmp.name)
        image_path = tmp.name

    # Run YOLO
    results = model(
        image_path,
        conf=CONF_THRESHOLD,
        imgsz=INFER_IMG_SIZE,
        verbose=False
    )

    # Read image for drawing
    annotated_img = cv2.imread(image_path)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    items = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            items.append(name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                annotated_img,
                name,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    counts = Counter(items)

    # ======================================================
    # RESULTS
    # ======================================================
    st.subheader("🔍 Original vs Detected")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original", width=350)

    with col2:
        st.image(annotated_img, caption="Detected", width=350)

    st.subheader("📊 Food Count Result")

    if not counts:
        st.warning("❌ No food items detected.")
    else:
        for item, qty in counts.items():
            st.write(f"**{item}** : {qty}")

        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Food Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.session_state.run_detect = False
