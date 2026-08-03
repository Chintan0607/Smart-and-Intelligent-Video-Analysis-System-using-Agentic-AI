import time

import requests
import streamlit as st

from config import NODE_A_PUBLIC_URL, UPLOAD_FOLDER
from utils import create_folders
from video_processor import save_uploaded_video

create_folders()

st.set_page_config(page_title="AI Surveillance Video Analyzer", layout="wide")

st.title("AI Surveillance Video Analyzer")
st.write("Upload a surveillance video to generate a quality-enhancement report.")

uploaded_video = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
    saved_path = save_uploaded_video(uploaded_video, UPLOAD_FOLDER)
    st.success("Video uploaded successfully!")

    if st.button("Generate Report", type="primary"):
        with st.spinner("Running agentic quality-enhancement pipeline..."):
            with open(saved_path, "rb") as f:
                upload_resp = requests.post(
                    f"{NODE_A_PUBLIC_URL}/agentic/upload",
                    files={"file": (uploaded_video.name, f, "video/mp4")},
                )
            upload_resp.raise_for_status()
            job_id = upload_resp.json()["job_id"]

            while True:
                status_resp = requests.get(f"{NODE_A_PUBLIC_URL}/status/{job_id}")
                status_resp.raise_for_status()
                job = status_resp.json()

                if job["status"] == "done":
                    break
                if job["status"] == "error":
                    st.error(f"Pipeline failed: {job.get('error_message', 'unknown error')}")
                    st.stop()

                time.sleep(2)

        pdf_resp = requests.get(f"{NODE_A_PUBLIC_URL}/report/{job_id}/pdf")
        pdf_resp.raise_for_status()

        st.session_state["report_pdf_bytes"] = pdf_resp.content
        st.session_state["report_pdf_name"] = f"{job_id}_report.pdf"

    if "report_pdf_bytes" in st.session_state:
        st.download_button(
            label="Download Report (PDF)",
            data=st.session_state["report_pdf_bytes"],
            file_name=st.session_state["report_pdf_name"],
            mime="application/pdf",
        )