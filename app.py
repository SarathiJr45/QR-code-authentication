import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("cnn_qr_model.h5")

def preprocess_image(image):
    image = np.array(image.convert("L"))  
    image = cv2.resize(image, (256, 256))  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=-1) 
    return image

st.title("QR Code Authentication")
st.write("Upload a QR code image to check if it's **First Print** or **Second Print**.")

uploaded_file = st.file_uploader("Choose a QR code image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded QR Code", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    class_names = ["First Print", "Second Print"]
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader(f"Prediction: **{predicted_class}**")
