"""
Restormer Worker (runs in .venv-gan, Python 3.13)
====================================================

Purpose
-------
Mirrors real_esrgan_worker.py — this is the real model backing the
"restormer" tool, replacing its stub. Uses the official Restormer
architecture (restormer_arch.py, sourced directly from
github.com/swz30/Restormer) with the "Real_Denoising" pretrained
checkpoint, since that's the task that matches this tool's role in the
pipeline: general restoration for heavy noise / compression artifacts.

Usage
-----
    python restormer_worker.py <input_image_path> <output_image_path>

Setup (one-time, in .venv-gan)
--------------------------------
1. pip install einops   (Restormer's only extra dependency beyond torch/cv2/numpy)
2. Place restormer_arch.py in the same folder as this script (provided
   alongside it — sourced directly from the official Restormer repo).
3. Download the Real_Denoising checkpoint into weights/:
       Invoke-WebRequest -Uri "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth" -OutFile "weights\\real_denoising.pth"

Notes
-----
- Restormer processes the whole image in one pass (no GAN-style tiling
  needed the way Real-ESRGAN requires) — but very large images can still
  hit VRAM limits on a 4GB card. This worker pads the input to a multiple
  of 8 (required by the model's downsampling) and, if you hit a CUDA OOM
  error, falls back to CPU automatically rather than crashing.
- No half-precision (fp16) is used, for the same reason as the
  Real-ESRGAN worker — GTX 16-series cards lack proper Tensor Core fp16
  support and will silently produce garbage/NaN output otherwise.
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from restormer_arch import Restormer

WEIGHTS_PATH = "weights/real_denoising.pth"


def load_model(device: str) -> Restormer:
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",  # required for the Real_Denoising checkpoint specifically
        dual_pixel_task=False,
    )
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint.get("params", checkpoint)  # official checkpoints wrap weights under "params"
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def run_inference(model: Restormer, image_bgr: np.ndarray, device: str) -> np.ndarray:
    # BGR (OpenCV) -> RGB, normalize to [0,1], HWC -> CHW -> add batch dim
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Restormer's downsampling path requires dimensions divisible by 8 — pad if needed
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        output = model(tensor_padded)

    output = output[:, :, :h, :w]  # crop back to original size
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    output_bgr = cv2.cvtColor((output * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return output_bgr


def main():
    if len(sys.argv) != 3:
        print("Usage: python restormer_worker.py <input_path> <output_path>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read input image: {input_path}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = load_model(device)
        try:
            output = run_inference(model, image, device)
        except torch.cuda.OutOfMemoryError:
            print("[restormer] CUDA out of memory — retrying on CPU (will be slower).", file=sys.stderr)
            torch.cuda.empty_cache()
            model.to("cpu")
            output = run_inference(model, image, "cpu")
    except Exception as exc:
        print(f"Restormer inference failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ok = cv2.imwrite(output_path, output)
    if not ok:
        print(f"Failed to write output image: {output_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
