"""
GET 324 - Laboratory Exercise 10 (Mini-Project)
Group CV20 - Psoriasis vs Eczema Binary Image Classification

Streamlit web application that loads a trained MobileNetV2 transfer-learning
model (eczema_psoriasis_model.h5) and classifies an uploaded skin image as
either Eczema or Psoriasis.

Run locally:  streamlit run app.py
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "eczema_psoriasis_model.keras"
IMG_SIZE = (224, 224)

# IMPORTANT: this order must match the class_names printed at the end of the
# training notebook (image_dataset_from_directory sorts folder names
# alphabetically -> Eczema = 0, Psoriasis = 1).
CLASS_NAMES = ["Eczema", "Psoriasis"]

st.set_page_config(
    page_title="CV20: Psoriasis vs Eczema Classifier",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    # MobileNetV2 preprocessing is baked into the saved model's first layers,
    # so we only resize/expand dims here.
    return array


def main():
    st.title("🩺 Psoriasis vs Eczema Classifier")
    st.caption("GET 324 Mini-Project — Group CV20, Department of Civil Engineering, University of Uyo")

    st.write(
        "Upload a close-up image of the affected skin area. The model will "
        "predict whether the image shows **Eczema** or **Psoriasis**."
    )

    st.warning(
        "⚠️ Educational tool only. This is a student coursework project and "
        "is **not** a medical diagnostic device. Please consult a qualified "
        "dermatologist for an actual diagnosis."
    )

    uploaded_file = st.file_uploader(
        "Choose a skin image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            try:
                model = load_model()
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)[0][0]

                predicted_index = int(prediction > 0.5)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = prediction if predicted_index == 1 else 1 - prediction

                st.success(f"**Prediction: {predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                st.progress(float(confidence))

            except FileNotFoundError:
                st.error(
                    f"Model file '{MODEL_PATH}' not found. Make sure it has "
                    "been uploaded to the same folder as app.py in the GitHub repo."
                )
            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")

    st.divider()
    st.caption(
        "Model: MobileNetV2 transfer learning (fine-tuned) · "
        "Dataset: Skin Diseases Image Dataset (Kaggle) — Eczema & Psoriasis classes only"
    )


if __name__ == "__main__":
    main()
