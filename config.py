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

SYSTEM_PROMPT = """You are an expert video quality analysis engine.

Analyze the provided sampled video frames for these technical flaws:
1. Motion blur, from none to severe.
2. Sensor noise or visible grain, from none to severe.
3. Compression artifacts such as blocking, ringing, banding, or mosquito noise.
4. The most suitable restoration/generative model class for fixing the dominant flaw
   (for example Real-ESRGAN, GFPGAN, SwinIR, CodeFormer, or DeblurGANv2).

OUTPUT REQUIREMENTS (STRICT):
- Respond in plain text only.
- Do NOT include markdown formatting, code fences, backticks, or language tags.
- Do NOT output JSON, dictionaries, arrays, or key-value syntax.
- Return exactly one sentence.
- Mention the main quality issue and the recommended restoration model in that sentence.
"""
