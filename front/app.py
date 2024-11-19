import os
import requests
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Simple Search Enginey", layout="wide")


# Get API details from environment variables, with defaults
API_HOST = os.getenv("API_HOST", "api")  # Default to 'api' for Docker
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

def truncate_text(text, max_length=50):
    return text[:max_length] + "..." if len(text) > max_length else text

def display_document(docs, selected_index, doc_type):
    st.write(f"{doc_type} Document:")
    st.write(docs[selected_index])

st.title("Simple Search Enginex")

col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader('Select image', ['jpg'])

st.write(image_file)

search_button = st.button("Submit")

# Initialize session state
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

if image_file:
    try:
        pil_image = Image.open(image_file)
        st.image(pil_image)
        files = {"image": image_file.getvalue()}
        response = requests.post(f"{API_URL}/process-image", files=files)
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
