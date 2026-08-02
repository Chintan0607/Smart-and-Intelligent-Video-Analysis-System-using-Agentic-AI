"""
agentic/local_enhancers.py

Real-ESRGAN lives on Node C (GPU) and is called over HTTP via the
existing _call_node_c(). These three tools have no remote worker yet,
so they run locally on Node A with OpenCV — cheap, no GPU needed.
Add Restormer/SwinIR/DeblurGAN later the same way Node C is wired,
not here.
"""

import os
import uuid

import cv2

from config import ENHANCED_FRAME_FOLDER

os.makedirs(ENHANCED_FRAME_FOLDER, exist_ok=True)


def _out_path(src_path: str, tag: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(ENHANCED_FRAME_FOLDER, f"{stem}_{tag}_{uuid.uuid4().hex[:6]}.jpg")


def opencv_denoise(image_path: str) -> str:
    img = cv2.imread(image_path)
    out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    out_path = _out_path(image_path, "denoise")
    cv2.imwrite(out_path, out)
    return out_path


def clahe(image_path: str) -> str:
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    out_path = _out_path(image_path, "clahe")
    cv2.imwrite(out_path, out)
    return out_path


def sharpen(image_path: str) -> str:
    img = cv2.imread(image_path)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    out = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    out_path = _out_path(image_path, "sharpen")
    cv2.imwrite(out_path, out)
    return out_path


LOCAL_TOOLS = {
    "opencv_denoise": opencv_denoise,
    "clahe": clahe,
    "sharpen": sharpen,
}
