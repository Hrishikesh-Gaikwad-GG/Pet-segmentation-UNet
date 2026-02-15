import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
from model import load_model
from utils import preprocess_image, predict_mask, create_overlay
import os
from pathlib import Path


# st.set_page_config(layout = 'wide')

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / 'samples'

st.title('🐱 Pet Segmentation Demo 🐶')
st.write('Upload as image of a pet and see segmentation results.')

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

mode = st.radio(
    "Choose input type:",
    ["Upload your pet Image", "Use Sample Image"]
)

image = None

if mode == 'Upload your pet Image':

    uploaded_file = st.file_uploader('Upload as image', type = ['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
else:

    sample_images = os.listdir(SAMPLE_DIR)
    image_path = image_select(
        'Select an Image',
        [SAMPLE_DIR / img for img in sample_images],
        use_container_width = False 
    )

    if image_path:
        image = Image.open(image_path).convert('RGB')

threshold = st.slider('Mask Threshold', 0.1, 0.9, 0.5, 0.05)

if image:
    image_np = np.array(image)

    input_tensor = preprocess_image(image)
    mask = predict_mask(model,image_np, input_tensor, threshold)
    overlay = create_overlay(image_np, mask)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Original')
        st.image(image_np)

    with col2:
        st.subheader('Predicted Mask')
        st.image(mask, clamp = True)

    with col3:
        st.subheader('Overlay')
        st.image(overlay)