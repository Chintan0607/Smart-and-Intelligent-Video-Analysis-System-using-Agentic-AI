import base64
import io
import os
import tempfile
from openai import OpenAI


import cv2
from PIL import Image
from huggingface_hub import InferenceClient

from config import (
    HF_API_TOKEN,
    VLM_CANDIDATES,
    HF_VLM_MODEL_ID_OVERRIDE,
    HF_PROVIDER_OVERRIDE,
    MAX_IMAGE_DIMENSION,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
)

MAX_VIDEO_FRAMES = 4


class VLMServiceError(Exception):
    """Raised when the VLM service fails to process or respond."""
    pass


def _preprocess_image(image_file) -> Image.Image:
    """
    Load, normalize color mode, and resize an image for optimal VLM throughput.

    Args:
        image_file: A file-like object (e.g. Streamlit UploadedFile) or path.

    Returns:
        A processed PIL.Image.Image instance.
    """
    try:
        image = Image.open(image_file)
    except Exception as exc:
        raise VLMServiceError(f"Unable to open image: {exc}") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(longest_side)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def _image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL suitable for chat-completion payloads."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _resize_image(image: Image.Image) -> Image.Image:
    """Resize an image so long video frames do not overload the VLM request."""
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(longest_side)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)
    return image


def _frame_to_image(frame) -> Image.Image:
    """Convert an OpenCV BGR frame into a resized PIL image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _resize_image(Image.fromarray(rgb_frame))


def _extract_video_frames(video_file, max_frames: int = MAX_VIDEO_FRAMES) -> list[Image.Image]:
    """Sample representative frames from a short uploaded video."""
    suffix = os.path.splitext(getattr(video_file, "name", ""))[1] or ".mp4"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_file.getbuffer())
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise VLMServiceError("Unable to open video file.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            step = max(frame_count // max_frames, 1)
            frame_indices = [min(index * step, frame_count - 1) for index in range(max_frames)]
        else:
            frame_indices = list(range(max_frames))

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if success and frame is not None:
                frames.append(_frame_to_image(frame))

        capture.release()

        if not frames:
            raise VLMServiceError("No readable frames could be extracted from the video.")

        return frames
    except VLMServiceError:
        raise
    except Exception as exc:
        raise VLMServiceError(f"Unable to process video: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _candidate_list():
    """Build the ordered list of (model, provider) pairs to attempt."""
    candidates = []
    if HF_VLM_MODEL_ID_OVERRIDE and HF_PROVIDER_OVERRIDE:
        candidates.append((HF_VLM_MODEL_ID_OVERRIDE, HF_PROVIDER_OVERRIDE))
    candidates.extend(VLM_CANDIDATES)
    return candidates


from openai import OpenAI

def _try_candidate(model_id: str, provider: str, data_urls: list[str], media_label: str):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_API_TOKEN,
        timeout=REQUEST_TIMEOUT,
    )
    user_content = [
        {
            "type": "text",
            "text": f"Analyze these sampled {media_label} frames for quality flaws in one sentence.",
        }
    ]
    user_content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for data_url in data_urls
    )

    completion = client.chat.completions.create(
        model=f"{model_id}:{provider}",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    return completion.choices[0].message.content

def analyze_image_flaws(image_file) -> str:
    """
    Run image quality analysis against a hosted vision-language model,
    automatically failing over across known-free (model, provider) pairs.

    Args:
        image_file: A file-like object containing the source image.

    Returns:
        The raw string content returned by the model (expected to be JSON).

    Raises:
        VLMServiceError: If preprocessing fails, no token is configured, or
            every candidate (model, provider) pair fails.
    """
    if not HF_API_TOKEN:
        raise VLMServiceError("Missing HF_API_TOKEN. Set it in your environment or .env file.")

    image = _preprocess_image(image_file)
    data_urls = [_image_to_data_url(image)]

    errors = []
    for model_id, provider in _candidate_list():
        try:
            content = _try_candidate(model_id, provider, data_urls, "image")
            if content and content.strip():
                return content.strip()
            errors.append(f"{model_id} via {provider}: empty response")
        except Exception as exc:
            errors.append(f"{model_id} via {provider}: {exc}")
            continue

    details = " | ".join(errors)
    raise VLMServiceError(
        f"All VLM candidates failed. Check https://huggingface.co/models?"
        f"inference_provider=hf-inference&pipeline_tag=image-text-to-text for live free models, "
        f"then set HF_VLM_MODEL_ID_OVERRIDE / HF_PROVIDER_OVERRIDE in .env. Details: {details}"
    )


def analyze_video_flaws(video_file) -> str:
    """
    Sample a short video and run quality analysis against the hosted VLM.

    Args:
        video_file: A file-like object containing the source video.

    Returns:
        The one-sentence text content returned by the model.

    Raises:
        VLMServiceError: If frame extraction fails, no token is configured, or
            every candidate (model, provider) pair fails.
    """
    if not HF_API_TOKEN:
        raise VLMServiceError("Missing HF_API_TOKEN. Set it in your environment or .env file.")

    frames = _extract_video_frames(video_file)
    data_urls = [_image_to_data_url(frame) for frame in frames]

    errors = []
    for model_id, provider in _candidate_list():
        try:
            content = _try_candidate(model_id, provider, data_urls, "video")
            if content and content.strip():
                return content.strip()
            errors.append(f"{model_id} via {provider}: empty response")
        except Exception as exc:
            errors.append(f"{model_id} via {provider}: {exc}")
            continue

    details = " | ".join(errors)
    raise VLMServiceError(
        f"All VLM candidates failed. Check https://huggingface.co/models?"
        f"inference_provider=hf-inference&pipeline_tag=image-text-to-text for live free models, "
        f"then set HF_VLM_MODEL_ID_OVERRIDE / HF_PROVIDER_OVERRIDE in .env. Details: {details}"
    )
