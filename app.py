import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.set_page_config(page_title="Medical Image Analyzer", page_icon="🩻", layout="centered")

@st.cache_resource
def load_model():
    # ImageNet-pretrained model used for a demo of image preprocessing and inference.
    # It is NOT trained or validated for medical diagnosis.
    return MobileNetV2(weights="imagenet")

st.title("🩻 Deep Learning Image Analyzer")
st.warning(
    "Educational demo only. This model is trained on general ImageNet images, not medical scans. "
    "Do not use its predictions for diagnosis, treatment, or clinical decisions."
)
st.write("Upload an image to demonstrate a deep-learning image-classification pipeline.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze image", type="primary"):
            with st.spinner("Loading model and analyzing image..."):
                model = load_model()
                resized = image.resize((224, 224))
                arr = np.asarray(resized, dtype=np.float32)
                batch = np.expand_dims(arr, axis=0)
                batch = preprocess_input(batch)
                predictions = model.predict(batch, verbose=0)
                decoded = decode_predictions(predictions, top=5)[0]
            st.subheader("Demo predictions (general ImageNet labels)")
            for _, label, score in decoded:
                st.write(f"**{label.replace('_', ' ')}** — {score * 100:.2f}%")
            st.info("These are general-object labels, not medical findings.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

with st.expander("How this demo works"):
    st.markdown(
        "1. Load an image and convert it to RGB.\\n"
        "2. Resize it to 224 × 224 pixels.\\n"
        "3. Apply MobileNetV2 preprocessing.\\n"
        "4. Run inference with ImageNet pretrained weights.\\n"
        "5. Display the top five general-image labels."
    )
