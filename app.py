import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚰 Water Meter Reading System MVP")
st.write("Upload an image of a water meter to detect the meter, window, and digits.")

# Load the model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose a water meter image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing detection...")
    
    # Run prediction
    results = model(image)
    
    # Plot results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detected Output", use_container_width=True)
