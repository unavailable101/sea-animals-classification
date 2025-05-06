import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import time

# Load icon
im = Image.open('./page_icon.png')

st.set_page_config(
    page_title="Sea Animals Classifier",
    page_icon=im,
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>

    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        font-family: 'Poppins', sans-serif !important;
        background-color: #0a1d37;
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #90e0ef;
    }

    label, .stFileUploader label {
        color: white !important;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.5em 1em;
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 10px;
    }

    .prediction {
        font-size: 1.5em;
        color: #00b4d8;
        text-align: center;
    }

    .image-preview {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    [data-testid="stFileUploader"] {
        border: 1px solid #1f4068;
        padding: 10px;
        border-radius: 5px;
    }

    .stButton>button {
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 1.2em;
    }

    .stSpinner > div {
        color: #ffffff;
    }

    /* Camera input button */
    [data-testid="stCameraInput"] button {
        color: white;
        font-weight: bold;
        border: 2px solid #90e0ef;
        border-radius: 8px;
    }

    /* Tabs customization */
    div[data-testid="stTabs"] {
        margin-top: 1rem;
        border-radius: 10px;
        overflow: hidden;
        background-color: white;
        padding: 0.5rem;
    }

    div[data-testid="stTabs"] button {
        color: #0a1d37 !important;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        margin: 0 0.3em;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 0 0 transparent;
    }

    div[data-testid="stTabs"] button:hover {
        background-color: #90e0ef;
        color: #003049 !important;
        transform: translateY(-1px);
    }

    div[data-testid="stTabs"] button:focus {
        animation: pulseTab 0.6s ease-in-out;
        background-color: #00b4d8;
        color: white !important;
        box-shadow: 0 0 0 4px rgba(0, 180, 216, 0.3);
    }

    /* Pulse animation */
    @keyframes pulseTab {
        0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(0, 180, 216, 0.5);
        }
        50% {
            transform: scale(1.03);
            box-shadow: 0 0 0 8px rgba(0, 180, 216, 0.1);
        }
        100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(0, 180, 216, 0);
        }
    }

    </style>
""", unsafe_allow_html=True)

st.title("Sea Animals Classifier")
st.markdown("""
Welcome to the Sea Animals Classifier.

This web app uses a deep learning model based on EfficientNetB0 to identify sea animals from images. It can classify images into 23 different categories, such as turtles, dolphins, sharks, squids, and others.

The model was trained using a dataset originally featured in a Kaggle notebook by [vencerlanz09](https://www.kaggle.com/code/vencerlanz09). We made a few modifications to improve it and adapt it for this app.

To get started, you can upload an image of a sea animal or take one using your camera. The model will try to predict which species it is.
""")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animals_classification_model_checkpoint.keras")
    with open("class_indices.pkl", "rb") as f:
        class_indices = pickle.load(f)
    return model, class_indices

model, class_indices = load_model()
class_names = [None] * len(class_indices)
for label, idx in class_indices.items():
    class_names[idx] = label

st.subheader("Upload or Take a Photo")
tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])

image = None
with tab1:
    uploaded_file = st.file_uploader("Upload a sea animal image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

with tab2:
    camera_file = st.camera_input("Take a photo")
    if camera_file:
        image = Image.open(camera_file).convert("RGB")

if image:
    st.markdown("<div class='image-preview'>", unsafe_allow_html=True)
    st.image(image, caption="Selected Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Progress bar simulation
    st.subheader("Processing...")
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    CONFIDENCE_THRESHOLD = 0.70

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Unknown"

            # Get top 3 predictions
            top_3_indices = prediction[0].argsort()[-3:][::-1]
            top_3 = [(class_names[i], prediction[0][i]) for i in top_3_indices]

            st.markdown(f"<div class='prediction'>Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
            st.markdown("<div class='confidence-badge'>Top 3 guesses:</div>", unsafe_allow_html=True)

            for label, conf in top_3:
                st.markdown(f"- {label}: **{conf*100:.2f}%**")
        else:
            predicted_class = class_names[predicted_class_idx]
            st.markdown(f"<div class='prediction'>Prediction: <strong>{predicted_class}</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-badge'>Confidence: {confidence*100:.2f}%</div>", unsafe_allow_html=True)

    # Plot all class probabilities
    st.subheader("Prediction Probabilities for All Classes")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, prediction[0], color='#00b4d8')
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontsize=9, color='white')
    st.pyplot(fig)

st.markdown("""
---
This app was built using TensorFlow and Streamlit.

The model is based on work by [vencerlanz09 on Kaggle](https://www.kaggle.com/code/vencerlanz09/sea-animals-classification-using-efficeintnetb7), and the dataset includes 23 sea animal classes such as clams, corals, crabs, dolphins, sharks, turtles, and more.
""")
