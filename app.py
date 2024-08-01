import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf
import pickle
from numpy.linalg import norm
from PIL import Image
from sklearn.neighbors import NearestNeighbors


st.write('# Fashion Recommendation System')
# Create the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

neig = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')

# Load features
file_names = pickle.load(open('filename.pkl', 'rb'))
feature_list = pickle.load(open('feature_list.pkl', 'rb'))
neig.fit(feature_list)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file buffer as a PIL image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Convert the PIL image to a format suitable for Keras
    img = img.resize((224, 224))  # Resize the image
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    processed_img = preprocess_input(img_array)

    # Predict using the model
    res = model.predict(processed_img).flatten()
    norm_res = res / norm(res)
    distance, indices = neig.kneighbors([norm_res])

    # Display the retrieved images in a column layout
    st.write("## Similar Images:")
    cols = st.columns(len(indices[0]))  # Create a column for each image
    for idx, col in zip(indices[0], cols):
        with col:
            st.image(file_names[int(idx)], use_column_width=True)
            
