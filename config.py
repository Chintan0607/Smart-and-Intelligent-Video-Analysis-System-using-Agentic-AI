# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

UPLOAD_FOLDER = "uploads"
EXTRACTED_FRAME_FOLDER = "extracted_frames"
ENHANCED_FRAME_FOLDER = "enhanced_frames"

NODE_A_HOST = "0.0.0.0"
NODE_A_PORT = 8000
NODE_A_PUBLIC_URL = "http://localhost:8000"

# NODE_B_URL = "http://192.168.1.11:8001"
# NODE_C_URL = "http://192.168.1.12:8002"

NODE_B_URL = "http://192.168.86.221:8001"
NODE_C_URL = "http://localhost:8002"

VLM_MODEL_QWEN = "Qwen/Qwen3-VL-2B-Instruct"
MAX_NEW_TOKENS = 384
QWEN_BATCH_SIZE = 4

REALESRGAN_MODEL_PATH = str(MODEL_CACHE_DIR / "Real-ESRGAN" / "RealESRGAN_x4plus.pth")
REALESRGAN_SCALE = 4
REALESRGAN_OUTSCALE = 2
DEVICE = "cuda"

HTTP_TIMEOUT_SECONDS = 120

MAX_ENHANCEMENT_RETRIES = 3
REPORT_FOLDER = "reports"