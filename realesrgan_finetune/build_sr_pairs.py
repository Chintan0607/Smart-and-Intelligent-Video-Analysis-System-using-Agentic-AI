"""
build_sr_pairs.py

For true 4x super-resolution training (matching RealESRGAN's architecture),
this creates a 4x-DOWNSCALED copy of each degraded image (blur/gaussian_noise/
low_light) and pairs it against the ORIGINAL-resolution clean GT image.

This is different from build_paired_meta.py (which pairs same-resolution
images for NAFNet-style restoration) -- this script actually resizes and
writes new image files, since the model needs to learn "small+degraded ->
large+clean".

Output structure:
    <root>/input_lq4x/<degradation>/<category>/<video_id>/frame_XXXXX.jpg
        (downscaled 4x copies -- new files, doesn't touch your original data)

Writes: meta_info_sr4x_<degradation>.txt
    Each line: <original-resolution GT path>, <4x-downscaled LQ path>

Usage:
    python build_sr_pairs.py --root /home/ai_user05/GAN-MODELS/GANN3/input1 --degradation blur
    python build_sr_pairs.py --root /home/ai_user05/GAN-MODELS/GANN3/input1 --degradation blur --limit 5000   # for a quick test run first
"""

import argparse
import os
import cv2
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                         help="Root folder containing input/ and clean_data/ (e.g. .../input1)")
    parser.add_argument("--degradation", required=True, choices=["blur", "gaussian_noise", "low_light"],
                         help="Which degradation type to build SR pairs for")
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor (matches RealESRGAN's 4x architecture)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N pairs (useful for a quick test run before committing to the full set)")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    gt_root = os.path.join(root, "clean_data", "ground_truth_output")
    deg_dir = os.path.join(root, "input", args.degradation)
    lq_out_root = os.path.join(root, "input_lq4x", args.degradation)

    if not os.path.isdir(gt_root):
        raise SystemExit(f"Ground truth folder not found: {gt_root}")
    if not os.path.isdir(deg_dir):
        raise SystemExit(f"Degradation folder not found: {deg_dir}")

    # collect all matching (gt_path, lq_source_path) pairs first
    pairs = []
    categories = sorted(d for d in os.listdir(deg_dir) if os.path.isdir(os.path.join(deg_dir, d)))
    for cat in categories:
        lq_cat_dir = os.path.join(deg_dir, cat)
        gt_cat_dir = os.path.join(gt_root, cat)
        if not os.path.isdir(gt_cat_dir):
            continue

        video_ids = sorted(d for d in os.listdir(lq_cat_dir) if os.path.isdir(os.path.join(lq_cat_dir, d)))
        for video_id in video_ids:
            lq_video_dir = os.path.join(lq_cat_dir, video_id)
            gt_video_dir = os.path.join(gt_cat_dir, video_id)
            if not os.path.isdir(gt_video_dir):
                continue

            for fname in sorted(os.listdir(lq_video_dir)):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                    continue
                lq_src = os.path.join(lq_video_dir, fname)
                gt_path = os.path.join(gt_video_dir, fname)
                if os.path.isfile(gt_path):
                    pairs.append((gt_path, lq_src, cat, video_id, fname))

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Found {len(pairs)} pairs to process for '{args.degradation}' (scale={args.scale}x downscale)")

    meta_lines = []
    skipped = 0

    for gt_path, lq_src, cat, video_id, fname in tqdm(pairs, desc=f"Downscaling {args.degradation}"):
        img = cv2.imread(lq_src, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        new_h, new_w = h // args.scale, w // args.scale
        if new_h < 1 or new_w < 1:
            skipped += 1
            continue

        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        out_dir = os.path.join(lq_out_root, cat, video_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, small)

        meta_lines.append(f"{gt_path}, {out_path}")

    meta_out = f"meta_info_sr4x_{args.degradation}.txt"
    with open(meta_out, "w") as f:
        f.write("\n".join(meta_lines) + "\n")

    print(f"\nDone. {len(meta_lines)} SR pairs written, {skipped} skipped (unreadable/too small).")
    print(f"Downscaled LQ images written under: {lq_out_root}")
    print(f"Meta info file: {os.path.abspath(meta_out)}")


if __name__ == "__main__":
    main()
