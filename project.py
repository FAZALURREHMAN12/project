import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os

# Load and preprocess data
@st.cache(allow_output_mutation=True)
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_data()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

model_path = "cifar10_cnn_model.h5"

@st.cache_resource
def get_model():
    if os.path.exists(model_path):
        # Load the model if it exists
        model = tf.keras.models.load_model(model_path)
        print("Model loaded from disk.")
    else:
        # Build and train the model if it doesn't exist
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(10),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.fit(
            train_images,
            train_labels,
            epochs=10,
            validation_data=(test_images, test_labels),
        )

        # Save the model
        model.save(model_path)
        print("Model saved to disk.")
    return model

model = get_model()

# Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("This app classifies images from the CIFAR-10 dataset.")

# Select an image from the test dataset
index = st.slider("Select an image index:", 0, len(test_images) - 1, 0)
selected_image = test_images[index]
true_label = class_names[test_labels[index][0]]

# Classify the selected image
def classify_image(image):
    img_array = tf.expand_dims(image, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class]

predicted_label = classify_image(selected_image)

# Display the image with its prediction
st.image(selected_image, caption=f"True Label: {true_label}", channels="RGB")
st.write(f"**Predicted Label:** {predicted_label}")
