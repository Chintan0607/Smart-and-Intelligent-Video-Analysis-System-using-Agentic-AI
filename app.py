import time  # 1. Import time module
import streamlit as st
from config import EXTRACTED_FRAME_FOLDER, FRAME_INTERVAL, UPLOAD_FOLDER
from services.qwen_vlm_service import QwenService
from utils import create_folders
from video_processor import (
    clear_previous_frames,
    extract_frames,
    get_video_metadata,
    save_uploaded_video,
)

create_folders()


@st.cache_resource
def load_vlm_service():
    return QwenService()


vlm = load_vlm_service()

st.set_page_config(page_title="AI Surveillance Video Analyzer", layout="wide")

st.sidebar.header("Processing Options")
frame_interval = st.sidebar.slider(
    "Extract every nth frame",
    min_value=1,
    max_value=100,
    value=FRAME_INTERVAL,
    step=1,
)

st.title("AI Surveillance Video Analyzer")
st.write("Upload a surveillance video to begin quality analysis.")

uploaded_video = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
    saved_path = save_uploaded_video(uploaded_video, UPLOAD_FOLDER)
    metadata = get_video_metadata(saved_path)

    st.success("Video uploaded successfully!")

    st.subheader("Video Information")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("FPS", f"{metadata['fps']:.2f}")
        st.metric("Frames", metadata["frame_count"])
        st.metric("Duration (s)", f"{metadata['duration']:.2f} sec")
    with col2:
        st.metric("Width", metadata["width"])
        st.metric("Height", metadata["height"])

    clear_previous_frames(EXTRACTED_FRAME_FOLDER)
    frames = extract_frames(saved_path, EXTRACTED_FRAME_FOLDER, frame_interval)
    st.info(f"Extracted {len(frames)} frames for temporal analysis.")

    if st.button("Run Quality Analysis Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        raw_responses = []
        frame_summaries = []
        total_frames = len(frames)

        # 2. Start execution timer
        start_time = time.time()

        for idx, frame in enumerate(frames):
            f_num = frame["frame_number"]
            status_text.text(
                f"Analyzing Frame {f_num} ({idx + 1}/{total_frames})..."
            )

            analysis = vlm.analyze_frame(frame["pil_image"])
            analysis["frame_number"] = f_num

            raw_responses.append(analysis)

            if "summary" in analysis and analysis["summary"]:
                frame_summaries.append(f"Frame {f_num}: {analysis['summary']}")

            progress_bar.progress((idx + 1) / total_frames)

        # 3. Calculate total elapsed processing time
        elapsed_time = time.time() - start_time
        status_text.text(f"Analysis complete in {elapsed_time:.2f} seconds!")

        aggregated_summary_text = " ".join(frame_summaries)

        # 4. Include processing time in payload for the downstream agent
        agent_payload = {
            "video_name": uploaded_video.name,
            "total_sampled_frames": total_frames,
            "processing_time_seconds": round(elapsed_time, 2),
            "aggregated_summary": aggregated_summary_text,
            "raw_frame_evaluations": raw_responses,
        }

        st.session_state["agent_payload"] = agent_payload

    # Display results and timing metrics
    if "agent_payload" in st.session_state:
        payload = st.session_state["agent_payload"]

        # Display performance metric alongside the summary
        st.subheader("Aggregated Video Quality Summary")
        m_col1, m_col2 = st.columns([1, 3])
        with m_col1:
            st.metric(
                "Pipeline Execution Time",
                f"{payload['processing_time_seconds']} s",
            )
        with m_col2:
            st.info(payload["aggregated_summary"])

        with st.expander("View Full Raw Payload (Agent Data)"):
            st.json(payload)

    st.subheader("Sample Extracted Frames Preview")
    preview_cols = st.columns(min(5, len(frames)))
    for idx, frame in enumerate(frames[:5]):
        with preview_cols[idx]:
            st.image(
                frame["rgb_image"],
                caption=f"Frame {frame['frame_number']}",
                use_container_width=True,
            )