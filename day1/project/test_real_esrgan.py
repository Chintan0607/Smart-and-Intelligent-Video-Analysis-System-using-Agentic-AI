"""
Quick standalone test: loads Real-ESRGAN and upscales one frame from the
test video. Run this in the .venv-gan environment to confirm the model
actually works end-to-end before it gets wired into the main pipeline.
"""
import sys
import time
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

if len(sys.argv) < 2:
    print("Usage: python test_real_esrgan.py <path_to_video_or_image>")
    sys.exit(1)

input_path = sys.argv[1]

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Grab one frame if a video was given, else load the image directly
if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
    cap = cv2.VideoCapture(input_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read a frame from the video.")
        sys.exit(1)
else:
    frame = cv2.imread(input_path)
    if frame is None:
        print("Could not read the image.")
        sys.exit(1)

print(f"Input frame shape: {frame.shape}")

print("Loading Real-ESRGAN model (RealESRGAN_x4plus)...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=model,
    tile=256,       # tiling keeps VRAM usage manageable on a 4GB card
    tile_pad=10,
    pre_pad=0,
    half=False,  # GTX 16-series (e.g. GTX 1650) lacks proper fp16 Tensor Core support —
                 # half=True produces NaN/all-zero output on this GPU generation. fp32 is slower
                 # but numerically correct.
)

print("Running inference...")
start = time.time()
output, _ = upsampler.enhance(frame, outscale=4)
elapsed = time.time() - start

print(f"Output frame shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Output min/max/mean: {output.min():.4f} / {output.max():.4f} / {output.mean():.4f}")
print(f"Inference time: {elapsed:.2f}s")

# Force a safe, explicit uint8 conversion regardless of what dtype/range
# the model returned — this is the most common cause of a "blank" saved
# image (imwrite doesn't auto-normalize floats, so out-of-range or
# 0.0-1.0-scaled data can save as solid black or solid white).
import numpy as np

if output.dtype != np.uint8:
    if output.max() <= 1.0:
        print("Output appears to be float in [0,1] range — scaling to 0-255.")
        output_to_save = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    else:
        print("Output appears to be float outside [0,1] — clipping to [0,255].")
        output_to_save = np.clip(output, 0, 255).astype(np.uint8)
else:
    output_to_save = output

cv2.imwrite("real_esrgan_test_output.png", output_to_save)
print("Saved result to real_esrgan_test_output.png — open it and compare against the input.")