"""
Step 1: Video Input
====================

Purpose
-------
This is the entry node of the LangGraph pipeline. It is responsible for:

1. Accepting a raw surveillance video path.
2. Validating that the file exists, is readable, and is a supported format.
3. Extracting technical metadata (fps, resolution, frame count, duration, codec).
4. Running cheap, non-DL heuristics to flag *likely* degradation types
   (low resolution, motion blur, low light, heavy noise/compression) so that
   Step 3 (VLM-based quality assessment) has a prior to work with instead of
   starting cold on every frame.
5. Populating a shared `PipelineState` object that every later LangGraph node
   (frame extraction, VLM assessment, ReAct agent, enhancement tools,
   validation loop, reconstruction, video understanding) reads from and
   writes to.

This module intentionally does NOT do frame extraction or enhancement —
that's Step 2 and Step 5. It only validates and characterizes the input.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import TypedDict, Optional, List, Dict, Any

# --------------------------------------------------------------------------
# 1. Shared pipeline state (grows as later steps are added)
# --------------------------------------------------------------------------
# This TypedDict is meant to be the LangGraph graph state. Each node
# (video_input_node, frame_extraction_node, quality_assessment_node, ...)
# reads/writes fields on it. Only the fields Step 1 owns are populated here;
# the rest exist as placeholders so downstream steps have a stable contract.

class PipelineState(TypedDict, total=False):
    # --- input ---
    video_path: str

    # --- populated by Step 1 ---
    video_id: str
    metadata: Dict[str, Any]
    degradation_flags: List[str]
    validation_status: str          # "valid" | "invalid"
    validation_errors: List[str]

    # --- placeholders for later steps ---
    frames_dir: str
    frame_paths: List[str]
    vlm_observations: List[Dict[str, Any]]
    enhancement_plan: List[Dict[str, Any]]
    reconstructed_video_path: str
    final_report: str
    error: Optional[str]


# --------------------------------------------------------------------------
# 2. Supported formats
# --------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

# Heuristic thresholds — tune against your actual camera fleet
MIN_ACCEPTABLE_WIDTH = 640
MIN_ACCEPTABLE_HEIGHT = 480
BLUR_VARIANCE_THRESHOLD = 100.0      # Laplacian variance below this ~= blurry
LOW_LIGHT_MEAN_THRESHOLD = 60.0      # mean pixel intensity (0-255) below this ~= poor lighting
NOISE_STD_THRESHOLD = 35.0           # rough proxy for heavy noise/compression artifacts


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    codec: str
    file_size_bytes: int
    sample_frames_checked: int = 0
    corrupted_frames_detected: int = 0


# --------------------------------------------------------------------------
# 3. Core validator / metadata extractor
# --------------------------------------------------------------------------

class VideoInputValidator:
    """Validates a surveillance video file and extracts metadata +
    preliminary degradation signals before it enters the enhancement
    pipeline."""

    def __init__(self, sample_frame_count: int = 8):
        # how many evenly-spaced frames to sample for the quick heuristics
        self.sample_frame_count = sample_frame_count

    def validate(self, video_path: str) -> tuple[bool, List[str]]:
        errors: List[str] = []

        if not video_path:
            return False, ["No video path provided."]

        if not os.path.isfile(video_path):
            return False, [f"File not found: {video_path}"]

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if os.path.getsize(video_path) == 0:
            errors.append("File is empty (0 bytes).")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            errors.append("OpenCV could not open the video (corrupt or unsupported codec).")
            cap.release()
            return False, errors

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            errors.append("Video reports zero or unreadable frame count.")

        ok, _ = cap.read()
        if not ok:
            errors.append("Could not read the first frame — file may be corrupted.")

        cap.release()
        return len(errors) == 0, errors

    def extract_metadata(self, video_path: str) -> VideoMetadata:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            chr(fourcc_int & 0xFF)
            + chr((fourcc_int >> 8) & 0xFF)
            + chr((fourcc_int >> 16) & 0xFF)
            + chr((fourcc_int >> 24) & 0xFF)
        ) if fourcc_int else "unknown"
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        return VideoMetadata(
            fps=round(fps, 3),
            frame_count=frame_count,
            width=width,
            height=height,
            duration_sec=round(duration, 2),
            codec=codec.strip(),
            file_size_bytes=os.path.getsize(video_path),
        )

    def assess_degradation(self, video_path: str, meta: VideoMetadata) -> List[str]:
        """Cheap, DL-free heuristics on a handful of sampled frames.
        These are *hints* for the VLM/ReAct agent in Step 3-4, not final
        decisions — the VLM still makes the authoritative call."""
        flags: List[str] = []

        if meta.width < MIN_ACCEPTABLE_WIDTH or meta.height < MIN_ACCEPTABLE_HEIGHT:
            flags.append("low_resolution")

        cap = cv2.VideoCapture(video_path)
        total = meta.frame_count or 1
        step = max(total // self.sample_frame_count, 1)

        blur_scores, brightness_scores, noise_scores = [], [], []
        checked, corrupted = 0, 0

        for i in range(self.sample_frame_count):
            frame_idx = min(i * step, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                corrupted += 1
                continue
            checked += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_scores.append(float(np.mean(gray)))
            noise_scores.append(float(np.std(gray)))

        cap.release()
        meta.sample_frames_checked = checked
        meta.corrupted_frames_detected = corrupted

        if corrupted > 0:
            flags.append("dropped_or_corrupted_frames")
        if blur_scores and (sum(blur_scores) / len(blur_scores)) < BLUR_VARIANCE_THRESHOLD:
            flags.append("motion_blur")
        if brightness_scores and (sum(brightness_scores) / len(brightness_scores)) < LOW_LIGHT_MEAN_THRESHOLD:
            flags.append("poor_lighting")
        if noise_scores and (sum(noise_scores) / len(noise_scores)) > NOISE_STD_THRESHOLD:
            flags.append("noise_or_compression_artifacts")

        return flags


# --------------------------------------------------------------------------
# 4. LangGraph node function
# --------------------------------------------------------------------------

def video_input_node(state: PipelineState) -> PipelineState:
    """Entry node of the LangGraph graph. Wire this in as:

        graph.add_node("video_input", video_input_node)
        graph.set_entry_point("video_input")
        graph.add_edge("video_input", "frame_extraction")   # Step 2

    Reads `state["video_path"]`, and on success populates:
        video_id, metadata, degradation_flags, validation_status
    On failure, sets validation_status="invalid", validation_errors, and
    `error`, so the graph can route to an error/end node instead of
    Step 2.
    """
    video_path = state.get("video_path", "")
    validator = VideoInputValidator()

    is_valid, errors = validator.validate(video_path)
    if not is_valid:
        return {
            **state,
            "validation_status": "invalid",
            "validation_errors": errors,
            "error": "; ".join(errors),
        }

    meta = validator.extract_metadata(video_path)
    flags = validator.assess_degradation(video_path, meta)
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    return {
        **state,
        "video_id": video_id,
        "metadata": {
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
            "duration_sec": meta.duration_sec,
            "codec": meta.codec,
            "file_size_bytes": meta.file_size_bytes,
            "sample_frames_checked": meta.sample_frames_checked,
            "corrupted_frames_detected": meta.corrupted_frames_detected,
        },
        "degradation_flags": flags,
        "validation_status": "valid",
        "validation_errors": [],
        "error": None,
    }


# --------------------------------------------------------------------------
# 5. Standalone smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python step1_video_input.py <path_to_video>")
        sys.exit(1)

    initial_state: PipelineState = {"video_path": sys.argv[1]}
    result_state = video_input_node(initial_state)

    print(json.dumps(
        {k: v for k, v in result_state.items() if k in
         ("video_id", "metadata", "degradation_flags", "validation_status", "validation_errors", "error")},
        indent=2,
    ))