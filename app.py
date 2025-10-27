import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
import tempfile

# ========== CONFIG ==========
MODEL_PATH = "waste_classifier_model1.h5"
TARGET_SIZE = (224, 224)
CLASSES = [
    "1-Cardboard",
    "2-Food Organics",
    "3-Glass",
    "4-Metal",
    "5-Miscellaneous Trash",
    "6-Paper",
    "7-Plastic",
    "8-Textile Trash",
    "9-Vegetation"
]

ADVISORY_TIPS = {
    "8-Textile Trash": "Dispose textiles responsibly. Donate or recycle when possible.",
    "1-Cardboard": "Flatten cardboard and keep it dry before sending to recycling.",
    "4-Metal": "Rinse metal objects and place them in the metal recycling bin.",
    "2-Food Organics": "Place in organic/compost bin. Avoid mixing with plastics.",
    "9-Vegetation": "Compost garden waste or send to organic recycling.",
    "7-Plastic": "Clean and sort plastics. Dispose in the plastic recycling bin.",
    "6-Paper": "Keep paper clean and dry before sending to recycling.",
    "5-Miscellaneous Trash": "Place in general waste bin. Avoid mixing with recyclables.",
    "3-Glass": "Handle glass carefully and place in the glass recycling bin."
}

BANNER_COLORS = {
    "8-Textile Trash": "#9b59b6",
    "1-Cardboard": "#d35400",
    "4-Metal": "#95a5a6",
    "2-Food Organics": "#27ae60",
    "9-Vegetation": "#2ecc71",
    "7-Plastic": "#f1c40f",
    "6-Paper": "#3498db",
    "5-Miscellaneous Trash": "#7f8c8d",
    "3-Glass": "#e74c3c"
}
# ===========================

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

def preprocess(image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, TARGET_SIZE, Image.Resampling.LANCZOS)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def tts_output(text):
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -------- STREAMLIT UI --------
st.title("♻️ EcoWatch — Waste Classifier (Upload Image)")
st.write("Upload a waste image and get classification + advisory output.")

model = load_model_cached()
file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        preprocessed = preprocess(img)
        prediction = model.predict(preprocessed)[0]
        class_idx = np.argmax(prediction)
        class_name = CLASSES[class_idx]
        confidence = float(prediction[class_idx]) * 100

    color = BANNER_COLORS[class_name]
    tip = ADVISORY_TIPS[class_name]

    st.markdown(
        f"<div style='background:{color};padding:15px;border-radius:8px;color:white'>"
        f"<h3>{class_name}</h3>"
        f"<p><b>Confidence:</b> {confidence:.2f}%</p>"
        f"<p><b>Advisory:</b> {tip}</p></div>",
        unsafe_allow_html=True
    )

    if st.button("🔊 Play Voice Advisory"):
        audio_path = tts_output(tip)
        audio_file = open(audio_path, "rb")
        st.audio(audio_file.read(), format="audio/mp3")

else:
    st.info("Please upload an image to begin classification.")
