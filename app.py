import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
# Load the pre-trained model
model1 = tf.keras.models.load_model('partly_trained3.h5')

# Define class names
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']



# Function to preprocess and make predictions
def predict_class(image):
    # Preprocess the image
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model1.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    return predicted_class

# Streamlit App
st.markdown('<h1>Image Classification App</h1>', unsafe_allow_html=True)
# First Tab
with st.sidebar:
    st.subheader('Upload Your Image')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image,width = 600)
    # Convert the image to numpy array
    img_array = np.array(image)

    # Make prediction
    prediction = predict_class(img_array)
    st.markdown("------------------------------------------------------")
    st.markdown(f'<p style="font-size:30px;color:while;"><strong>Prediction : </strong> {prediction}</p>', unsafe_allow_html=True)
    st.markdown("------------------------------------------------------")
# Second Tab
if st.sidebar.checkbox('Show Images'):
    arr = ['Data Augmentation : ', 'Model : ', "Confusion Matrix : ","Accuracy : ","Conclusion : ","Dataset : "]
    for i in range(1, 7):
        head = arr[i-1]
        st.header(f'<p style="font-size:30px;color:while;text-align: center;"><strong>{head} : </strong></p>', unsafe_allow_html=True)
        image_path = f"{i}.jpeg"
        image = Image.open(image_path)
        # Centered image using HTML and CSS
        st.image(image,width = 600)
