# This file will hold all the settings that might change later on
 

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

UPLOAD_FOLDER = "uploads"
EXTRACTED_FRAME_FOLDER = "extracted_frames"
ENHANCED_FRAME_FOLDER = "enhanced_frames"
OUTPUT_FOLDER = "outputs"

FRAME_INTERVAL = 30 

SUPPORTED_VIDEO_FORMATS = ["mp4","avi","mov","mkv"]

VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_MODEL_QWEN = "Qwen/Qwen3-VL-2B-Instruct"

DEVICE = "cuda"

MAX_NEW_TOKENS = 384

DO_SAMPLE = False

REALESRGAN_MODEL_PATH = "/home/varad/ML_Workspace/proj_test/models/huggingface/Real-ESRGAN/RealESRGAN_x4plus.pth"
REALESRGAN_SCALE = 4
REALESRGAN_OUTSCALE = 2