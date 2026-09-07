import os
import time
import pickle
from pathlib import Path

import cv2
import gdown
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🧠 Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_PATH = Path("model.pkl")

# Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "1i5jKiPQ6VGX9F0S3ZCz35YbQ2P1ZXYui"


@st.cache_resource
def load_model():

    """
    Download model from Google Drive only once,
    then load the pickle model.
    """

    if not MODEL_PATH.exists():

        with st.spinner("Downloading AI model..."):

            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

            gdown.download(
                url,
                str(MODEL_PATH),
                quiet=False
            )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model file could not be downloaded."
        )

    if MODEL_PATH.stat().st_size == 0:

        raise ValueError(
            "Downloaded model file is empty."
        )

    with open(MODEL_PATH, "rb") as file:

        model = pickle.load(file)

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(uploaded_file):

    """
    Convert uploaded MRI image into model input.

    IMPORTANT:
    This follows your original code:
    Resize -> Normalize -> Flatten
    """

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    resized = cv2.resize(
        image_array,
        (224, 224)
    )

    normalized = resized.astype(np.float32) / 255.0

    # Original code used flattening
    model_input = normalized.reshape(1, -1)

    return image, resized, model_input


# =========================================================
# PREDICTION
# =========================================================

def get_prediction(model, model_input):

    prediction = int(
        np.asarray(
            model.predict(model_input)
        ).ravel()[0]
    )

    if hasattr(model, "predict_proba"):

        probabilities = np.asarray(
            model.predict_proba(model_input)
        )[0]

    elif hasattr(model, "decision_function"):

        score = float(
            np.asarray(
                model.decision_function(model_input)
            ).ravel()[0]
        )

        tumor_probability = 1 / (1 + np.exp(-score))

        probabilities = np.array([
            1 - tumor_probability,
            tumor_probability
        ])

    else:

        probabilities = np.array([
            0.0,
            1.0
        ]) if prediction == 1 else np.array([
            1.0,
            0.0
        ])

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = probabilities / probabilities.sum()

    return prediction, probabilities


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f0c29 0%,
            #302b63 50%,
            #24243e 100%
        );
    }

    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton > button {
        border-radius: 25px;
        font-weight: bold;
    }

    .image-container {
        border-radius: 15px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div style="text-align:center; padding:25px 0;">

        <h1>🧠 Brain Tumor Detection System</h1>

        <p style="font-size:1.2rem; color:#a0a0a0;">
            AI-assisted MRI image analysis
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as error:

    st.error("The AI model could not be loaded.")

    st.exception(error)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        0.5,
        0.05
    )

    st.info(
        "Upload an MRI image and click Analyze Scan."
    )

    st.caption(
        "Supported formats: JPG, JPEG, PNG"
    )


# =========================================================
# MAIN UI
# =========================================================

left, right = st.columns([2, 1])


# =========================================================
# UPLOAD IMAGE
# =========================================================

with left:

    st.subheader("📤 Upload MRI Scan")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image, resized_image, model_input = preprocess_image(
                uploaded_file
            )

            st.image(
                image,
                caption="Uploaded MRI Scan",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                "The uploaded image could not be processed."
            )

            st.exception(error)

            st.stop()


# =========================================================
# PREDICTION PANEL
# =========================================================

with right:

    st.subheader("🧠 Prediction Panel")

    if uploaded_file is None:

        st.info(
            "Please upload an MRI image first."
        )

    elif st.button(
        "🔍 Analyze Scan",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner(
            "🤖 AI is analyzing the MRI scan..."
        ):

            time.sleep(0.5)

            try:

                prediction, probabilities = get_prediction(
                    model,
                    model_input
                )

            except Exception as error:

                st.error(
                    "Prediction failed. The model may expect "
                    "a different input shape or preprocessing method."
                )

                st.exception(error)

                st.stop()

        confidence = float(
            np.max(probabilities)
        )

        no_tumor_probability = float(
            probabilities[0]
        )

        tumor_probability = float(
            probabilities[1]
        )

        st.divider()

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.error("⚠️ Tumor Detected")

            st.warning(
                "This result is AI-assisted and is not a "
                "medical diagnosis. Please consult a "
                "qualified medical professional."
            )

        else:

            st.success("✅ No Tumor Detected")

            st.info(
                "This result is AI-assisted and is not a "
                "medical diagnosis. A medical professional "
                "should review the scan."
            )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Confidence Score",
                f"{confidence * 100:.1f}%"
            )

        with col_b:

            if confidence < confidence_threshold:

                st.warning("⚠️ Low Confidence")

            else:

                st.success("✓ High Confidence")

        # -------------------------------------------------
        # CHART
        # -------------------------------------------------

        chart = px.bar(
            x=["No Tumor", "Tumor"],
            y=[
                no_tumor_probability,
                tumor_probability
            ],
            labels={
                "x": "Class",
                "y": "Probability"
            },
            title="Prediction Probability"
        )

        chart.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------

        report = (
            "MRI Analysis Report\n\n"

            f"Prediction: "
            f"{'Tumor Detected' if prediction == 1 else 'No Tumor Detected'}\n"

            f"Confidence: "
            f"{confidence * 100:.1f}%\n"

            f"No Tumor Probability: "
            f"{no_tumor_probability * 100:.1f}%\n"

            f"Tumor Probability: "
            f"{tumor_probability * 100:.1f}%\n"
        )

        st.download_button(
            "📥 Download Report",
            data=report,
            file_name="mri_report.txt",
            mime="text/plain"
        )


# =========================================================
# IMAGE PROCESSING PREVIEW
# =========================================================

if uploaded_file is not None:

    st.divider()

    st.subheader("🧪 Image Processing Preview")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

    with c2:

        gray = cv2.cvtColor(
            resized_image,
            cv2.COLOR_RGB2GRAY
        )

        st.image(
            gray,
            caption="Grayscale",
            use_container_width=True
        )

    with c3:

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        st.image(
            edges,
            caption="Edge Detection",
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ This application is for educational and research "
    "purposes only. It must not replace professional "
    "medical advice."
)
