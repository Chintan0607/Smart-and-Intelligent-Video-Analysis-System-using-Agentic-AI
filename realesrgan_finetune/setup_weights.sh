#!/bin/bash
# setup_weights.sh
#
# Downloads the pretrained RealESRGAN_x4plus.pth generator weights to
# fine-tune from. Sourced directly from the official Real-ESRGAN GitHub
# release (no GitHub Release upload needed on your end for this one,
# since it's small and hosted officially).
#
# Usage:
#   bash setup_weights.sh

set -e

mkdir -p weights weights_nafnet

# RealESRGAN pretrained generator (official GitHub release)
RE_PATH="weights/RealESRGAN_x4plus.pth"
RE_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
if [ -f "$RE_PATH" ]; then
    echo "[skip] already exists: $RE_PATH"
else
    echo "[download] RealESRGAN_x4plus.pth -> $RE_PATH"
    curl -L -o "$RE_PATH" "$RE_URL"
fi

# NAFNet pretrained checkpoints -- reusing the same assets already uploaded
# to your GANN3 repo's release (public URLs, reusable across repos)
GANN3_RELEASE="https://github.com/Pratiksonkusare/GANN3/releases/download/v1.0"

NAF_GOPRO="weights_nafnet/NAFNet-GoPro-width64.pth"
if [ -f "$NAF_GOPRO" ]; then
    echo "[skip] already exists: $NAF_GOPRO"
else
    echo "[download] NAFNet-GoPro-width64.pth -> $NAF_GOPRO"
    curl -L -o "$NAF_GOPRO" "${GANN3_RELEASE}/NAFNet-GoPro-width64.pth"
fi

NAF_SIDD="weights_nafnet/NAFNet-SIDD-width64.pth"
if [ -f "$NAF_SIDD" ]; then
    echo "[skip] already exists: $NAF_SIDD"
else
    echo "[download] NAFNet-SIDD-width64.pth -> $NAF_SIDD"
    curl -L -o "$NAF_SIDD" "${GANN3_RELEASE}/NAFNet-SIDD-width64.pth"
fi

echo "Done."
