import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

VLM_CANDIDATES = [
    (os.getenv("HF_VLM_MODEL_ID_1", "Qwen/Qwen2.5-VL-7B-Instruct"), "hf-inference"),
    (os.getenv("HF_VLM_MODEL_ID_2", "meta-llama/Llama-3.2-11B-Vision-Instruct"), "hf-inference"),
    (os.getenv("HF_VLM_MODEL_ID_3", "Qwen/Qwen2.5-VL-7B-Instruct"), "auto"),
]

HF_VLM_MODEL_ID_OVERRIDE = os.getenv("HF_VLM_MODEL_ID_OVERRIDE", "")
HF_PROVIDER_OVERRIDE = os.getenv("HF_PROVIDER_OVERRIDE", "")

MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1024"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

SYSTEM_PROMPT = """You are an expert image quality analysis engine.

Analyze the provided image strictly for the following technical flaws:
1. motion_blur: a float between 0.0 (none) and 1.0 (severe) representing the degree of motion blur.
2. sensor_noise: a float between 0.0 (none) and 1.0 (severe) representing visible sensor/grain noise.
3. compression_artifacts: a float between 0.0 (none) and 1.0 (severe) representing blocking, ringing, or other compression artifacts.
4. recommended_gan: a string naming the most suitable restoration/generative model class for fixing the dominant flaw (e.g. "Real-ESRGAN", "GFPGAN", "SwinIR", "CodeFormer", "DeblurGANv2").

OUTPUT REQUIREMENTS (STRICT):
- Respond with ONLY a single valid JSON object and nothing else.
- Do NOT include markdown formatting, code fences, backticks, or language tags.
- Do NOT include any explanation, preamble, commentary, or trailing text.
- Do NOT wrap the JSON in any other structure.
- The JSON object MUST match exactly this schema and key order:

{"motion_blur": float, "sensor_noise": float, "compression_artifacts": float, "recommended_gan": string}

Any deviation from this format is considered a failed response."""