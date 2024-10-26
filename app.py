import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('fire_detector.h5')

# Set up Streamlit UI
st.title("Wildfire Detection from Satellite Images")
st.write("Upload an image to check if it detects fire or not.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img2 = cv2.imdecode(file_bytes, 1)
    test_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(test_img2, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for model prediction
    test_img2 = cv2.resize(test_img2, (256, 256))
    test_img2 = test_img2.reshape((1, 256, 256, 3))
    test_img2 = test_img2 / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(test_img2)[0][0]  # Assuming sigmoid activation in the output layer
    
    # Interpret the prediction
    if prediction >= 0.5:
        result = "No Fire"
    else:
        result = "Fire"

    # Display the result
    st.write(f"Prediction: {result} (Probability: {prediction:.2f})")
