import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Load precomputed features and file list
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

# Initialize the model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

st.title('Clothing Recommender System')

def save_file(uploaded_file):
    """ Saves uploaded file to disk. """
    upload_dir = "uploader"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    try:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {str(e)}")
        return None

def extract_img_features(img_path, model):
    """ Extract image features using the ResNet50 model. """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # Normalize the result
    result_normalized = flatten_result / norm(flatten_result)
    return result_normalized

def recommend(features, features_list):
    """ Recommend images based on nearest neighbors. """
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

def display_images(indices):
    """ Display recommended images. """
    for i, idx in enumerate(indices[0]):
        try:
            img_path = img_files_list[idx]
            image = Image.open(img_path)
            st.image(image, caption=f"Recommended Image {i+1}", width=200)
        except Exception as e:
            st.error(f"Failed to display image {img_files_list[idx]}: {str(e)}")

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    file_path = save_file(uploaded_file)
    if file_path:
        try:
            show_image = Image.open(file_path)
            st.image(show_image, caption="Uploaded Image", width=400)
            features = extract_img_features(file_path, model)
            img_indices = recommend(features, features_list)
            display_images(img_indices)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Failed to upload or save the file.")

