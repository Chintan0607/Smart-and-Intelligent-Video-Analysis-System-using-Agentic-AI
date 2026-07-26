"""
Step 2: Frame Extraction
=========================

Purpose
-------
Takes the validated video from Step 1 and extracts frames into memory
(no disk writes) at a configurable sampling rate — by default ~1 frame per
second, since surveillance footage rarely needs every single frame
analyzed for scene-understanding purposes, and in-memory downsampling
keeps Step 3 (VLM quality assessment) fast during development/testing.

Each extracted frame is wrapped in a `FrameRecord` carrying:
    - the numpy array (BGR, as read by OpenCV)
    - its original frame index in the source video
    - its timestamp in seconds

These records go into `state["frames"]`, which Step 3 (VLM quality
assessment) and Step 5 (enhancement tools) will consume and mutate
in place (e.g. replacing `.image` with an enhanced version) without
ever touching disk until Step 7 (reconstruction).

Note: in-memory storage is fine for short clips and testing. For long
or high-resolution surveillance footage this will grow the LangGraph
state significantly — swap `store_mode="disk"` back in when moving to
production-scale videos.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import TypedDict, Optional, List, Dict, Any

from step1_video_input import PipelineState  # reuse the shared state contract


# --------------------------------------------------------------------------
# 1. Frame record — the unit passed between nodes
# --------------------------------------------------------------------------

@dataclass
class FrameRecord:
    frame_index: int          # index in the ORIGINAL video (not the sampled sequence)
    timestamp_sec: float
    image: np.ndarray         # BGR uint8 array, shape (H, W, 3)
    enhancement_history: List[str] = field(default_factory=list)  # tools applied so far
    quality_notes: Optional[str] = None  # filled in by Step 3 (VLM)


# Extend the shared pipeline state with what Step 2 owns.
# (PipelineState already reserves "frame_paths" as a placeholder from Step 1;
#  since we're going in-memory we instead use "frames" for FrameRecord objects
#  and leave frame_paths empty/unused for this run.)
class PipelineStateWithFrames(PipelineState, total=False):
    frames: List[FrameRecord]
    sampling_fps: float
    frames_extracted: int
    frames_sampled: int


# --------------------------------------------------------------------------
# 2. Extractor
# --------------------------------------------------------------------------

class FrameExtractor:
    """Extracts frames from a validated video into memory at a target
    sampling rate (frames per second of *video time*, not source fps)."""

    def __init__(self, target_fps: float = 1.0):
        self.target_fps = target_fps

    def extract(self, video_path: str, source_fps: float, frame_count: int) -> List[FrameRecord]:
        if source_fps <= 0:
            raise ValueError("source_fps must be > 0 to compute a sampling interval.")

        # how many source frames to skip between samples
        interval = max(int(round(source_fps / self.target_fps)), 1)

        cap = cv2.VideoCapture(video_path)
        records: List[FrameRecord] = []

        for frame_idx in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue  # skip unreadable frames rather than fail the whole extraction
            timestamp = frame_idx / source_fps
            records.append(FrameRecord(
                frame_index=frame_idx,
                timestamp_sec=round(timestamp, 3),
                image=frame,
            ))

        cap.release()
        return records


# --------------------------------------------------------------------------
# 3. LangGraph node function
# --------------------------------------------------------------------------

def frame_extraction_node(state: PipelineStateWithFrames) -> PipelineStateWithFrames:
    """Second node in the graph. Wire it in as:

        graph.add_node("frame_extraction", frame_extraction_node)
        graph.add_edge("video_input", "frame_extraction")
        graph.add_edge("frame_extraction", "quality_assessment")   # Step 3

    Requires state["validation_status"] == "valid" (Step 1 must have
    succeeded) and state["metadata"] to be populated.
    """
    if state.get("validation_status") != "valid":
        return {
            **state,
            "error": state.get("error") or "frame_extraction skipped: video failed Step 1 validation.",
        }

    meta = state.get("metadata", {})
    source_fps = meta.get("fps", 0.0)
    frame_count = meta.get("frame_count", 0)
    video_path = state.get("video_path", "")

    target_fps = state.get("sampling_fps", 1.0)  # default: 1 frame per second
    extractor = FrameExtractor(target_fps=target_fps)

    try:
        frames = extractor.extract(video_path, source_fps, frame_count)
    except Exception as exc:
        return {**state, "error": f"frame_extraction failed: {exc}"}

    return {
        **state,
        "frames": frames,
        "sampling_fps": target_fps,
        "frames_extracted": frame_count,
        "frames_sampled": len(frames),
        "error": None,
    }


# --------------------------------------------------------------------------
# 4. Standalone smoke test — chains Step 1 -> Step 2
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node

    if len(sys.argv) < 2:
        print("Usage: python step2_frame_extraction.py <path_to_video> [target_fps]")
        sys.exit(1)

    target_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    state: PipelineStateWithFrames = {
        "video_path": sys.argv[1],
        "sampling_fps": target_fps,
    }
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)

    print(f"Source frames: {state['frames_extracted']}")
    print(f"Sampled at:    {state['sampling_fps']} fps")
    print(f"Frames kept:   {state['frames_sampled']}")
    for fr in state["frames"][:5]:
        print(f"  idx={fr.frame_index:5d}  t={fr.timestamp_sec:6.2f}s  shape={fr.image.shape}")
    if state["frames_sampled"] > 5:
        print(f"  ... ({state['frames_sampled'] - 5} more)")
