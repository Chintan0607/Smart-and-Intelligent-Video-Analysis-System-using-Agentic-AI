import streamlit as st

from config import UPLOAD_FOLDER
from utils import create_folders
from video_processor import save_uploaded_video

# Create folders when the application starts
create_folders()

st.set_page_config(
    page_title="AI Surveillance Video Analyzer",
    layout="wide"
)

st.title("AI Surveillance Video Analyzer")

st.write("Upload a surveillance video to begin processing.")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
    saved_path = save_uploaded_video(
        uploaded_video,
        UPLOAD_FOLDER
    )

    st.success("Video uploaded successfully!")

    st.video(saved_path)

    st.write("Saved at: ")

    st.code(saved_path)