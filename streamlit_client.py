import time

import requests
import streamlit as st

NODE_A_URL = "http://localhost:8000"

# Extraction and validation are visually distinct phases with different
# "denominators" (frames found so far vs. frames fully validated out of
# a now-known total) — split the bar so it never looks stuck at 0% or
# jumps discontinuously when the phase changes.
EXTRACTION_PROGRESS_SHARE = 0.3
PROCESSING_PROGRESS_SHARE = 1.0 - EXTRACTION_PROGRESS_SHARE

st.set_page_config(page_title="Agentic Video Analyzer", layout="centered")
st.title("Agentic Surveillance Video Analyzer")
st.write(
    "Upload a video file. The autonomous agent will extract keyframes, route them "
    "through the VLM validation loop, apply targeted tools (OpenCV/GAN), "
    "and compile a comprehensive PDF report."
)

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None and st.button("Run Agentic Pipeline", type="primary"):
    files = {"file": (uploaded_video.name, uploaded_video.getvalue())}

    with st.spinner("Uploading video to Agentic Orchestrator..."):
        upload_response = requests.post(f"{NODE_A_URL}/agentic/upload", files=files)
        upload_response.raise_for_status()
        job_id = upload_response.json()["job_id"]

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    job = {}

    while True:
        status_response = requests.get(f"{NODE_A_URL}/agentic/status/{job_id}")
        status_response.raise_for_status()
        job = status_response.json()

        frames_extracted = job.get("frames_extracted", 0)
        completed = job.get("frames_completed", 0)
        total = job.get("total_frames_estimate")

        if job["status"] == "extracting":
            # Total keyframe count isn't known yet during extraction —
            # nudge the bar forward as frames come in instead of parking
            # it at a flat placeholder value.
            progress = min(0.02 * frames_extracted, EXTRACTION_PROGRESS_SHARE)
            status_text.text(
                f"Status: EXTRACTING KEYFRAMES | Frames found so far: {frames_extracted}"
            )
        elif job["status"] == "processing":
            fraction_done = (completed / total) if total else 0.0
            progress = EXTRACTION_PROGRESS_SHARE + PROCESSING_PROGRESS_SHARE * fraction_done
            status_text.text(
                f"Status: RUNNING AGENT VALIDATION LOOPS | "
                f"Keyframes extracted: {frames_extracted} | "
                f"Frames fully validated: {completed}/{total or '?'}"
            )
        elif job["status"] == "done":
            progress = 1.0
            status_text.text(
                f"Status: DONE | Keyframes extracted: {frames_extracted} | "
                f"Frames fully validated: {completed}/{total or completed}"
            )
        else:
            progress = 0.0
            status_text.text(f"Status: {job['status'].upper()}")

        progress_bar.progress(progress)

        if job["status"] in ("done", "error"):
            break
        time.sleep(2)

    if job["status"] == "error":
        st.error(f"Pipeline execution failed: {job.get('error_message', 'Unknown error')}")
    else:
        st.success("Agentic analysis and report generation complete.")

        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Keyframes Extracted", job.get("frames_extracted", 0))
        summary_col2.metric("Frames Fully Validated", job.get("frames_completed", 0))

        with st.spinner("Retrieving PDF report..."):
            pdf_response = requests.get(f"{NODE_A_URL}/agentic/report/{job_id}/pdf")
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content

        st.download_button(
            label="📄 Download Agentic PDF Report",
            data=pdf_bytes,
            file_name=f"{uploaded_video.name}_agent_report.pdf",
            mime="application/pdf",
            type="primary",
        )