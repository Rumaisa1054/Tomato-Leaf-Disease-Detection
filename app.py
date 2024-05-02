import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model1 = tf.keras.models.load_model('partly_trained3.h5')

# Function to make predictions
def predict_class(image):
    # Preprocess the image
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model1.predict(image)
    return prediction

# Streamlit App
st.title('Image Classification App')

# First Tab
with st.sidebar:
    st.subheader('Upload Your Image')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to numpy array
    img_array = np.array(image)

    # Make prediction
    prediction = predict_class(img_array)
    st.write('Prediction:', prediction)

# Second Tab
if st.sidebar.checkbox('Show Images'):
    st.subheader('Images from 1.jpeg to 6.jpeg')
    for i in range(1, 7):
        image_path = f"{i}.jpeg"
        image = Image.open(image_path)
        st.image(image, caption=f'Image {i}', use_column_width=True)
