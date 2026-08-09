# RealESRGAN Fine-Tuning — Enhancement Module (Node C)

Supervised fine-tuning of Real-ESRGAN on real paired CCTV data, producing the
generator weights used by the Node C enhancement service.

**Author:** Pratik Sonkusare
**Project:** Smart and Intelligent Video Analysis System using Agentic AI —
Group 5, PGCP-AI, C-DAC ACTS Pune

---

## Why this exists

Node C originally ran stock Real-ESRGAN weights. These handle generic upscaling
well, but did not recover our blur cases — the degradations in real CCTV footage
don't match the synthetic degradation model Real-ESRGAN was trained on. Rather
than accept that, the generator was fine-tuned in a fully supervised way on real
degraded/clean frame pairs from our own dataset.

## Data pipeline

| Script | Purpose |
|---|---|
| `build_paired_meta.py` | Matches degraded frames against clean ground truth by `category / video_id / frame filename`, writes a meta_info index per degradation type (blur, gaussian_noise, low_light). |
| `build_sr_pairs.py` | Writes 4x-downscaled copies of the degraded frames, so pairs match Real-ESRGAN's 4x architecture (small+degraded → large+clean). Does not modify the source data. |

~210K blur pairs were used for the super-resolution run.

## Training

Two stages, on PARAM Rudra (NVIDIA A100, MIG partition).

**Stage 1 — PSNR-oriented** (`options/finetune_realesrgan_sr4x.yml`)
L1 loss only, 20,000 iterations, initialised from official
`RealESRGAN_x4plus.pth` (`params_ema`). Produces a stable generator before
adversarial training.

**Stage 2 — adversarial** (`options/finetune_realesrgan_gan.yml`)
`SRGANModel` with a `UNetDiscriminatorSN`, initialised from the stage-1
generator. 60,000 iterations. Loss = L1 + VGG19 perceptual + vanilla GAN at
weight 0.1, so pixel and perceptual terms still dominate and the adversarial
term only pushes texture realism.

Shared settings: RRDBNet generator (23 blocks, 64 feat, 16.7M params),
Adam @ 1e-4, MultiStepLR (milestones 10k/20k, γ=0.5), EMA 0.999.

NAFNet configs (`finetune_nafnet_*.yml`) are included as the same-resolution
restoration alternative that was evaluated alongside the SR approach.

## Implementation note

The obvious choice, `RealESRGANModel`, is hard-wired to `RealESRGANDataset`'s
*synthetic* degradation pipeline and fails on real paired data. Switching to
basicsr's generic `SRGANModel` with `RealESRGANPairedDataset` is what made
supervised training on real pairs work at all. This is the single non-obvious
piece of this module.

## Weights

Trained weights are not committed here (128 MB). Download from the release:

**https://github.com/Pratiksonkusare/GAN4/releases**

`net_g_latest.pth` contains both `params` and `params_ema` — use `params_ema`
for inference.

## Setup

```bash
bash setup.sh                                    # clones training repos, installs deps
python build_paired_meta.py --root <path>/input1 --out meta_info_paired.txt
python build_sr_pairs.py --root <path>/input1 --degradation blur
cd RealESRGAN-train
python realesrgan/train.py -opt ../options/finetune_realesrgan_sr4x.yml
```
