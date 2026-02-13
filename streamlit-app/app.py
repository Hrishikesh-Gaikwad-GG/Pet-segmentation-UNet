import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import load_model
from utils import preprocess_image, predict_mask, create_overlay


st.set_page_config(layout = 'wide')

st.title('Pet Segmentation Demo')
st.write('Upload as image of a pet and see segmentation results.')

model = load_model()

uploaded_file = st.file_uploader('Upload as image', type = ['jpg', 'png'])

threshold = st.slider('Mask Threshold', 0.1, 0.9, 0.5, 0.05)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
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