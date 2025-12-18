import streamlit as st
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import tempfile
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64
from streamlit_paste_button import paste_image_button

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/user171125/Documents/model testing/best copy 4.pt"
CONF_THRESHOLD = 0.25

st.set_page_config(page_title="Food Detection", layout="centered")
st.title("🍽️ Food Detection App")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- STATE ----------------
if "image" not in st.session_state:
    st.session_state.image = None

if "run_detect" not in st.session_state:
    st.session_state.run_detect = False

# ---------------- INPUTS ----------------
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

# ---------------- LOAD IMAGE ----------------
if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

elif image_url:

    try:
        # 🔥 CASE 1: BASE64 DATA URL
        if image_url.startswith("data:image"):
            header, encoded = image_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            st.session_state.image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # CASE 2: NORMAL HTTP URL
        else:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            st.session_state.image = Image.open(BytesIO(r.content)).convert("RGB")

    except Exception as e:
        st.error("Could not load image from URL")

elif paste_result.image_data is not None:
    # paste-button returns bytes OR PIL
    if isinstance(paste_result.image_data, bytes):
        st.session_state.image = Image.open(
            BytesIO(paste_result.image_data)
        ).convert("RGB")
    else:
        st.session_state.image = paste_result.image_data.convert("RGB")

# ---------------- BUTTON ----------------
if st.button("🚀 Start Detecting"):
    if st.session_state.image is None:
        st.warning("⚠️ Please upload / paste an image first.")
    else:
        st.session_state.run_detect = True

# ---------------- YOLO DETECTION ----------------
if st.session_state.run_detect:

    image = st.session_state.image

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    results = model(image_path, conf=CONF_THRESHOLD)

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

    # ---------------- RESULTS ----------------
    st.subheader("🔍 Original vs Detected")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", width=350)

    with col2:
        st.image(annotated_img, caption="Detected", width=350)

    st.subheader("📊 Food Count Result")

    if not counts:
        st.warning("No food items detected.")
    else:
        for item, qty in counts.items():
            st.write(f"**{item}** : {qty}")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Food Count")
        plt.xticks(rotation=45)
     

    st.session_state.run_detect = False
