import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚰 Water Meter Reading System MVP")
st.write("Upload an image of a water meter to detect and read the final digits.")

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
    boxes = results[0].boxes
    
    # Filter for digits (classes 0 to 9 match class names '0' to '9')
    digit_detections = []
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Check if the detected class is one of our digits
        if class_name in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            x_min = box.xyxy[0][0].item() # Get the horizontal position
            digit_detections.append((x_min, class_name))
            
    # Sort detections from left to right based on their X coordinate
    digit_detections.sort(key=lambda x: x[0])
    
    # Join the sorted digits together to form the final reading
    final_reading = "".join([digit[1] for digit in digit_detections])
    
    # Display the final reading prominently!
    if final_reading:
        st.success(f"### 🔢 Detected Water Meter Reading: **{final_reading}**")
    else:
        st.warning("### ⚠️ No digits detected clearly. Try another image!")
        
    # Plot results to show visual boxes too
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Visual Detections", use_container_width=True)
