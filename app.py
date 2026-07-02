import json

import streamlit as st

from vlm_service import analyze_image_flaws, VLMServiceError

st.set_page_config(page_title="Image Quality Analyzer", layout="centered")


def _strip_code_fences(raw: str) -> str:
    """Defensively remove markdown code fences in case the model ignores instructions."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_response(raw: str) -> dict:
    cleaned = _strip_code_fences(raw)
    return json.loads(cleaned)


def main():
    st.title("Image Quality Analyzer")
    st.caption("Upload an image to detect motion blur, sensor noise, and compression artifacts.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    run_analysis = st.button("Run Analysis", type="primary", disabled=uploaded_file is None)

    if run_analysis and uploaded_file is not None:
        with st.spinner("Analyzing image quality..."):
            try:
                raw_response = analyze_image_flaws(uploaded_file)
            except VLMServiceError as exc:
                st.error(f"Analysis failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected error during analysis: {exc}")
                return

            try:
                result = _parse_response(raw_response)
            except json.JSONDecodeError:
                st.error("The model returned a response that could not be parsed as JSON.")
                with st.expander("View raw model output"):
                    st.text(raw_response)
                return

            required_keys = {"motion_blur", "sensor_noise", "compression_artifacts", "recommended_gan"}
            missing_keys = required_keys - result.keys()
            if missing_keys:
                st.warning(f"Response is missing expected fields: {', '.join(sorted(missing_keys))}")

            st.subheader("Analysis Result")
            st.json(result)


if __name__ == "__main__":
    main()