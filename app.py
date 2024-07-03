import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    return keras.models.load_model('fashion_mnist_cnn_model.h5')

model = load_model()

# Class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Preprocess image
def preprocess_image(image):
    # Resize image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Reshape for model input
    image = image.reshape((1, 28, 28, 1))
    return image

# Main Streamlit app
def main():
    st.title("Fashion MNIST Classifier")
    st.write("Upload an image of a fashion item to classify it!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        class_name = class_labels[class_index]
        confidence = float(prediction[0][class_index])

        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()