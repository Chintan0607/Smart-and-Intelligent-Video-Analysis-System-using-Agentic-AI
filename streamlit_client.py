# streamlit_client.py
import time

import requests
import streamlit as st

NODE_A_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Surveillance Video Analyzer", layout="wide")
st.title("AI Surveillance Video Analyzer (Distributed)")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None and st.button("Run Pipeline", type="primary"):
    files = {"file": (uploaded_video.name, uploaded_video.getvalue())}
    upload_response = requests.post(f"{NODE_A_URL}/upload", files=files)
    upload_response.raise_for_status()
    job_id = upload_response.json()["job_id"]

    progress_bar = st.progress(0)
    status_text = st.empty()
    job = {}

    while True:
        status_response = requests.get(f"{NODE_A_URL}/status/{job_id}")
        status_response.raise_for_status()
        job = status_response.json()

        total = job.get("total_frames_estimate") or max(job["frames_extracted"], 1)
        progress_bar.progress(min(job["frames_completed"] / total, 1.0))
        status_text.text(
            f"Status: {job['status']} | Extracted: {job['frames_extracted']} | "
            f"Completed: {job['frames_completed']}"
        )

        if job["status"] in ("done", "error"):
            break
        time.sleep(2)

    if job["status"] == "error":
        st.error(job.get("error_message", "Unknown error"))
    else:
        report_response = requests.get(f"{NODE_A_URL}/report/{job_id}")
        report_response.raise_for_status()
        report = report_response.json()

        st.subheader("Video Report")
        col1, col2 = st.columns(2)
        col1.metric("Processing Time (s)", report["processing_time_seconds"])
        col2.metric("Keyframes Extracted", report["total_keyframes_extracted"])

        for entry in report["entries"]:
            f_col, vlm_col, e_col = st.columns(3)
            with f_col:
                st.image(entry["frame"]["url"], caption=f"Frame {entry['frame']['frame_number']}")
            with vlm_col:
                st.json(entry["vlm"])
            with e_col:
                if entry["enhancement"] and entry["enhancement"]["applied"]:
                    st.image(entry["enhancement"]["enhanced_url"], caption="Enhanced")