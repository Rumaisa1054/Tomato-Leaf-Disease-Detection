import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model1 = tf.keras.models.load_model('partly_trained3.h5')

# Define class names
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

# Define image size
image_size = 224

# Function to preprocess and make predictions
def predict_class(image):
    # Preprocess the image
    image = tf.image.resize(image, (image_size, image_size))
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model1.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

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
    pred = 'Prediction :' + prediction
    st.header(pred)

# Second Tab
if st.sidebar.checkbox('Show Images'):
    st.subheader('Images from 1.jpeg to 6.jpeg')
    arr = ['Data Augmentation : ', 'Model : ', "Confusion Matrix : ","Accuracy : ","Conclusion : ","Dataset : "]
    for i in range(1, 7):
        head = arr[i]
        st.header(head)
        image_path = f"{i}.jpeg"
        image = Image.open(image_path)
        st.image(image, caption=f'Image {i}', use_column_width=True)
