#!/bin/bash
# setup.sh
#
# One-shot setup for the RealESRGAN fine-tuning repo. Clones the official
# training repo, installs dependencies, and downloads the pretrained
# generator weights to fine-tune from.
#
# Usage (from inside the freshly cloned repo folder):
#   bash setup.sh

set -e

echo "=================================================="
echo " RealESRGAN fine-tuning -- environment setup"
echo "=================================================="
echo ""

# --- 1. Install top-level Python dependencies -------------------------------
echo "[1/4] Installing base dependencies..."
pip install -r requirements.txt
echo ""

# --- 2. Clone training repos -------------------------------------------------
echo "[2/4] Cloning training repos..."
if [ -d "RealESRGAN-train" ]; then
    echo "  [skip] RealESRGAN-train already exists"
else
    git clone https://github.com/xinntao/Real-ESRGAN.git RealESRGAN-train
fi

if [ -d "NAFNet-train" ]; then
    echo "  [skip] NAFNet-train already exists"
else
    git clone https://github.com/megvii-research/NAFNet.git NAFNet-train
fi
echo ""

# --- 3. Install requirements -------------------------------------------------
echo "[3/4] Installing RealESRGAN-train requirements..."
pip install -r RealESRGAN-train/requirements.txt
(cd RealESRGAN-train && python setup.py develop)

echo "  [info] NAFNet-train ships its own bundled basicsr -- run its"
echo "         basicsr/train.py directly (not via 'setup.py develop'),"
echo "         same as we had to do to avoid the basicsr shadowing issue."
echo ""

# --- 4. Download pretrained weights to fine-tune from ------------------------
echo "[4/4] Downloading pretrained weights..."
bash setup_weights.sh
echo ""

echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. python build_paired_meta.py --root <path to your input1 folder> --out meta_info_paired.txt"
echo "  2. For NAFNet (recommended -- correct same-resolution architecture):"
echo "       cd NAFNet-train"
echo "       python basicsr/train.py -opt ../options/finetune_nafnet_blur.yml"
echo "       python basicsr/train.py -opt ../options/finetune_nafnet_noise.yml"
