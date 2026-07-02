import base64
import io

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


def _candidate_list():
    """Build the ordered list of (model, provider) pairs to attempt."""
    candidates = []
    if HF_VLM_MODEL_ID_OVERRIDE and HF_PROVIDER_OVERRIDE:
        candidates.append((HF_VLM_MODEL_ID_OVERRIDE, HF_PROVIDER_OVERRIDE))
    candidates.extend(VLM_CANDIDATES)
    return candidates


def _try_candidate(model_id: str, provider: str, data_url: str):
    """Attempt a single (model, provider) call. Returns content string or raises."""
    client = InferenceClient(
        provider=provider,
        api_key=HF_API_TOKEN,
        timeout=REQUEST_TIMEOUT,
    )
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image for quality flaws."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=512,
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
    data_url = _image_to_data_url(image)

    errors = []
    for model_id, provider in _candidate_list():
        try:
            content = _try_candidate(model_id, provider, data_url)
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