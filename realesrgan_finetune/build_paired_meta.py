"""
build_paired_meta.py

Matches your real degraded images (blur/, gaussian_noise/, low_light/) against
their clean counterparts (clean_data/ground_truth_output/) by category +
video_id + frame filename, and writes a paired meta_info file for
Real-ESRGAN's paired training dataset (RealESRGANPairedDataset).

Expects this structure (matches your actual Rudra layout):

    <root>/
      input/
        blur/<category>/<video_id>/frame_XXXXX.jpg
        gaussian_noise/<category>/<video_id>/frame_XXXXX.jpg
        low_light/<category>/<video_id>/frame_XXXXX.jpg
      clean_data/
        ground_truth_output/<category>/<video_id>/frame_XXXXX.jpg

Each line written to the output file is:
    <absolute_gt_path>, <absolute_lq_path>

Usage:
    python build_paired_meta.py --root /home/ai_user05/GAN-MODELS/GANN3/input1 --out meta_info_paired.txt
"""

import argparse
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
DEGRADATION_TYPES = ["blur", "gaussian_noise", "low_light"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                         help="Root folder containing input/ and clean_data/ (e.g. .../input1)")
    parser.add_argument("--out", default="meta_info_paired.txt")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    gt_root = os.path.join(root, "clean_data", "ground_truth_output")
    input_root = os.path.join(root, "input")

    if not os.path.isdir(gt_root):
        raise SystemExit(f"Ground truth folder not found: {gt_root}")
    if not os.path.isdir(input_root):
        raise SystemExit(f"Input folder not found: {input_root}")

    matched = 0
    missing_gt = 0
    lines_by_degradation = {d: [] for d in DEGRADATION_TYPES}

    for degradation in DEGRADATION_TYPES:
        deg_dir = os.path.join(input_root, degradation)
        if not os.path.isdir(deg_dir):
            print(f"[skip] {degradation}/ not found under {input_root}")
            continue

        categories = sorted(
            d for d in os.listdir(deg_dir)
            if os.path.isdir(os.path.join(deg_dir, d))
        )
        print(f"\n[{degradation}] categories found: {categories}")

        for cat in categories:
            lq_cat_dir = os.path.join(deg_dir, cat)
            gt_cat_dir = os.path.join(gt_root, cat)

            if not os.path.isdir(gt_cat_dir):
                print(f"  [warn] no matching GT category folder for '{cat}' -- skipping this category")
                continue

            video_ids = sorted(
                d for d in os.listdir(lq_cat_dir)
                if os.path.isdir(os.path.join(lq_cat_dir, d))
            )

            for video_id in video_ids:
                lq_video_dir = os.path.join(lq_cat_dir, video_id)
                gt_video_dir = os.path.join(gt_cat_dir, video_id)

                if not os.path.isdir(gt_video_dir):
                    print(f"    [warn] no matching GT video folder for '{cat}/{video_id}' -- skipping")
                    continue

                for fname in sorted(os.listdir(lq_video_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                        continue  # skips frames_metadata.csv etc.

                    lq_path = os.path.join(lq_video_dir, fname)
                    gt_path = os.path.join(gt_video_dir, fname)  # assumes matching frame filename

                    if os.path.isfile(gt_path):
                        lines_by_degradation[degradation].append(f"{gt_path}, {lq_path}")
                        matched += 1
                    else:
                        missing_gt += 1

    print(f"\n{'='*50}")
    for degradation, lines in lines_by_degradation.items():
        out_path = f"meta_info_{degradation}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[{degradation}] {len(lines)} pairs -> {os.path.abspath(out_path)}")

    # also write the combined file for convenience / backward compatibility
    all_lines = [l for lines in lines_by_degradation.values() for l in lines]
    with open(args.out, "w") as f:
        f.write("\n".join(all_lines) + "\n")

    print(f"\nTotal matched pairs: {matched}")
    print(f"Degraded images with NO matching GT filename: {missing_gt}")
    print(f"Combined file: {os.path.abspath(args.out)}")
    if missing_gt > 0:
        print("\n[note] Some degraded frames had no same-named file in the matching")
        print("       ground_truth_output/<category>/<video_id>/ folder.")
        print("       Check a few examples manually if this number looks unexpectedly high.")


if __name__ == "__main__":
    main()

