from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

UPLOAD_FOLDER = "uploads"
EXTRACTED_FRAME_FOLDER = "extracted_frames"
ENHANCED_FRAME_FOLDER = "enhanced_frames"

# --------------------------------------------------------------------------
# Agentic pipeline settings
# --------------------------------------------------------------------------
AGENTIC_FRAME_FOLDER = "agentic_frames"
MAX_AGENT_ITERATIONS = 3
AGENT_FRAME_CONCURRENCY = 1
PDF_REPORT_FOLDER = "reports"

# --------------------------------------------------------------------------
# Node addressing
# --------------------------------------------------------------------------
NODE_A_HOST = "0.0.0.0"
NODE_A_PORT = 8000
NODE_A_PUBLIC_URL = "http://localhost:8000"

NODE_B_URL = "http://192.168.86.221:8001"
NODE_C_URL = "http://localhost:8002"

HTTP_TIMEOUT_SECONDS = 420

# --------------------------------------------------------------------------
# Qwen VLM (Node B)
# --------------------------------------------------------------------------
VLM_MODEL_QWEN = "Qwen/Qwen3-VL-2B-Instruct"
MAX_NEW_TOKENS = 384
QWEN_BATCH_SIZE = 4

# --------------------------------------------------------------------------
# Real-ESRGAN (Node C)
# --------------------------------------------------------------------------
REALESRGAN_MODEL_PATH = str(MODEL_CACHE_DIR / "Real-ESRGAN" / "RealESRGAN_x4plus.pth")
REALESRGAN_SCALE = 4
REALESRGAN_OUTSCALE = 2
DEVICE = "cuda"