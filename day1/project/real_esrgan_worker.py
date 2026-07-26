"""
Real-ESRGAN Worker (runs in .venv-gan, Python 3.13)
======================================================

Purpose
-------
The main pipeline runs in a Python 3.14 venv (.venv) because that's what
the project was built in. Real-ESRGAN's dependency chain (torch+CUDA,
basicsr, realesrgan) requires Python 3.13 or earlier for CUDA wheel
support — hence the separate .venv-gan environment.

Since two different Python environments can't directly import each
other's packages, this script is a standalone worker: it takes an input
image path and an output image path as command-line arguments, runs
Real-ESRGAN on the input, and writes the result to the output path. The
main pipeline (in step5_tool_execution.py) calls this via subprocess,
using temp files to pass the frame in and get the enhanced frame back.

Usage
-----
    python real_esrgan_worker.py <input_image_path> <output_image_path>

Exits 0 on success, non-zero on failure (with an error message on
stderr) — the calling subprocess wrapper in Step 5 checks this exit code
to decide whether to fall back to the stub.

Notes
-----
- half=False is REQUIRED on GTX 16-series cards (e.g. GTX 1650) — fp16
  produces NaN/all-zero output on this GPU generation due to lack of
  proper Tensor Core fp16 support. If you're running this on an RTX card
  or better, you can set half=True for a significant speed improvement.
- The model is loaded fresh on every subprocess call (no persistent
  server), so there's a fixed model-load overhead per call in addition
  to inference time. For batch processing many frames, consider
  extending this into a small persistent local HTTP server instead —
  left as a future optimization, not needed for this project's scale.
"""

import sys
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

WEIGHTS_PATH = "weights/RealESRGAN_x4plus.pth"
TILE_SIZE = 256  # keeps VRAM usage manageable on 4GB cards; increase if you have more VRAM


def main():
    if len(sys.argv) != 3:
        print("Usage: python real_esrgan_worker.py <input_path> <output_path>", file=sys.stderr)
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read input image: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=WEIGHTS_PATH,
            model=model,
            tile=TILE_SIZE,
            tile_pad=10,
            pre_pad=0,
            half=False,  # see module docstring — required on GTX 16-series
        )
        output, _ = upsampler.enhance(image, outscale=4)
    except Exception as exc:
        print(f"Real-ESRGAN inference failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ok = cv2.imwrite(output_path, output)
    if not ok:
        print(f"Failed to write output image: {output_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
