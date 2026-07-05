import streamlit as st

from vlm_service import VLMServiceError, analyze_video_flaws

st.set_page_config(page_title="Video Quality Analyzer", layout="centered")

VIDEO_FORMATS = ["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"]


def _one_sentence(raw_response: str) -> str:
    """Keep the UI plain even if the model returns extra whitespace or lines."""
    text = " ".join(raw_response.strip().split())
    if not text:
        return "No analysis result was returned."
    return text


def _video_format(uploaded_file) -> str:
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if extension == "mp4":
        return "video/mp4"
    if extension == "webm":
        return "video/webm"
    if extension in {"mpeg", "mpg"}:
        return "video/mpeg"
    if extension == "mov":
        return "video/quicktime"
    if extension == "avi":
        return "video/x-msvideo"
    if extension == "mkv":
        return "video/x-matroska"
    return "video/mp4"


def main():
    st.title("Video Quality Analyzer")
    st.caption("Upload a short 10-15 second video to detect visible quality flaws.")

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=VIDEO_FORMATS,
    )

    if uploaded_file is not None:
        video_bytes = uploaded_file.getvalue()
        st.video(video_bytes, format=_video_format(uploaded_file))

    run_analysis = st.button("Run Analysis", type="primary", disabled=uploaded_file is None)

    if run_analysis and uploaded_file is not None:
        with st.spinner("Analyzing video quality..."):
            try:
                raw_response = analyze_video_flaws(uploaded_file)
            except VLMServiceError as exc:
                st.error(f"Analysis failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected error during analysis: {exc}")
                return

        st.subheader("Analysis Result")
        st.write(_one_sentence(raw_response))


if __name__ == "__main__":
    main()
