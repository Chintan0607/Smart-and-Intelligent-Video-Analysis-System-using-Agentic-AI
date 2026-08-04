"""
opencv_tools.py

Local, dependency-light enhancement tools the Agent can apply without a
round-trip to Node C. These target defects Real-ESRGAN is a poor fit
for: it's trained to hallucinate plausible high-frequency detail from a
low-res input, which does very little for *motion blur* (the detail
exists, it's just smeared) or *exposure/contrast* problems (the pixels
are fine, they're just poorly distributed).

Every function is pure: numpy array in, numpy array out, no file I/O,
no HTTP. That keeps them unit-testable and keeps the agent loop (which
owns file naming and persistence) as the only place that touches disk.
"""
from __future__ import annotations

import cv2
import numpy as np


def unsharp_mask(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0,
) -> np.ndarray:
    """
    Classic unsharp masking: blur the image, subtract the blur from the
    original to isolate high-frequency edge content, then add that back
    in, amplified. Good first-line response to mild-to-moderate motion
    blur and general softness.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        low_contrast_mask = np.absolute(image.astype(int) - blurred.astype(int)) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Laplacian-kernel sharpening. Stronger and more edge-aggressive than
    unsharp masking — reach for this when unsharp masking wasn't enough
    on a previous iteration (i.e. the VLM still flags blur after one
    unsharp pass) since it responds better to more pronounced blur.
    """
    gray_laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    sharpened = image.astype(np.float64) - strength * gray_laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization, applied to the L
    channel in LAB space so color isn't distorted. This is the right
    tool for uneven low light (e.g. a shadowed corner of an otherwise
    fine frame) — plain global histogram equalization tends to blow out
    already-bright regions in the same frame.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization on the luminance channel (YCrCb space).
    Simpler and faster than CLAHE, but can over-brighten regions that
    were already well-lit — use as a fallback when CLAHE alone hasn't
    resolved a flagged low-light/contrast defect after one iteration.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    y_equalized = cv2.equalizeHist(y_channel)

    merged = cv2.merge((y_equalized, cr_channel, cb_channel))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# --------------------------------------------------------------------------
# Dispatch table used by the agent's router (see agent_orchestrator.decide_tool)
# --------------------------------------------------------------------------
from schemas import ToolType  # noqa: E402  (after functions, to keep this file readable top-down)

_LOCAL_TOOL_FUNCS = {
    ToolType.OPENCV_UNSHARP: unsharp_mask,
    ToolType.OPENCV_LAPLACIAN: laplacian_sharpen,
    ToolType.OPENCV_CLAHE: apply_clahe,
}


def apply_local_tool(image: np.ndarray, tool: ToolType) -> np.ndarray:
    """Single entry point the agent loop calls for any non-GAN tool."""
    func = _LOCAL_TOOL_FUNCS.get(tool)
    if func is None:
        raise ValueError(f"{tool} is not a local OpenCV tool (route to Node C instead)")
    return func(image)
