import os
import requests
import streamlit as st
from PIL import Image

# Get API details from environment variables, with defaults
API_HOST = os.getenv("API_HOST", "api")  # Default to 'api' for Docker
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

def generate_caption(image_file):
    try:
        files = {"image": image_file.getvalue()}
        with st.spinner('Generating caption...'):
            response = requests.post(f"{API_URL}/process-image", files=files)
        if response.ok:
            caption = response.json()["caption"]
            st.write(f"Caption: {caption}")
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

def upload_file(image_file, caption):
    try:
        files = {"image": ("image.jpg", image_file.getvalue(), image_file.type)}
        data = {"caption": caption}
        with st.spinner('Uploading file...'):
            response = requests.post(f"{API_URL}/upload-image", files=files, data=data)
        if response.ok:
            st.write(f"Uploaded")
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

st.set_page_config(page_title="Image caption generator", layout="wide")
st.title("Image caption generator")

col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader('', ['jpg', 'png'])
    if image_file:
        st.image(image_file)
        st.write('Submit your image to generate a caption')
        if st.button("Generate caption"):
            if image_file:
                generate_caption(image_file)

        st.write("Alternatively, help us out by uploading an image with a caption, for training our model")
        caption = st.text_input('Your caption')
        if caption and st.button("Submit caption"):
            upload_file(image_file, caption)

