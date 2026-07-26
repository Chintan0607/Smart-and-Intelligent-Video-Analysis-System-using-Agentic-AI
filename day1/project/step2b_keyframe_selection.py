"""
Step 2b: Intelligent Keyframe Selection
==========================================

Purpose
-------
Addresses project objective 2.2 (Intelligent Keyframe Extraction): Step 2
samples frames at a fixed interval (~1fps) regardless of content — this
node sits right after it and discards frames that are near-duplicates of
the last KEPT frame, so downstream VLM calls (Step 3, Step 8) aren't
wasted re-assessing/re-captioning a scene that hasn't actually changed.

Method
------
Deterministic, no VLM call needed (keeps it fast and free):

1. Convert each candidate frame to grayscale.
2. Compute the mean absolute pixel difference against the last KEPT
   frame (not the immediately-previous frame — this prevents slow drift
   through many small changes from silently skipping past a real change).
3. If the difference exceeds `change_threshold`, keep the frame as a new
   keyframe. Otherwise, discard it as redundant.
4. The first and last sampled frames are always kept, regardless of
   similarity, so the clip's start/end context is never lost.

This is a heuristic, not a learned redundancy model — `change_threshold`
is the one knob to tune per camera/scene. Default (8.0 on a 0-255 scale)
is a reasonable starting point: real motion/lighting changes typically
score well above it, sensor noise/compression jitter typically stays
below it. Tighten it if too much is being kept; loosen it if genuine
changes are getting discarded.

Note on "empty frame" filtering: this same mechanism naturally handles
long static stretches (an empty room with nothing happening) — since
every frame is compared to the last KEPT one, a long run of unchanging
frames only produces ONE keyframe until something actually changes,
rather than N redundant "empty room" frames.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord

DEFAULT_CHANGE_THRESHOLD = 8.0


@dataclass
class KeyframeSelectionSummary:
    frames_in: int
    frames_kept: int
    frames_discarded: int
    change_threshold: float


class KeyframeSelector:
    def __init__(self, change_threshold: float = DEFAULT_CHANGE_THRESHOLD):
        self.change_threshold = change_threshold

    @staticmethod
    def _diff_score(a: np.ndarray, b: np.ndarray) -> float:
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if gray_a.shape != gray_b.shape:
            gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
        return float(np.mean(cv2.absdiff(gray_a, gray_b)))

    def select(self, frames: List[FrameRecord]) -> tuple[List[FrameRecord], KeyframeSelectionSummary]:
        if not frames:
            return [], KeyframeSelectionSummary(0, 0, 0, self.change_threshold)

        ordered = sorted(frames, key=lambda f: f.frame_index)
        kept: List[FrameRecord] = [ordered[0]]  # always keep the first frame

        for i in range(1, len(ordered) - 1):  # last frame handled separately below
            candidate = ordered[i]
            last_kept = kept[-1]
            score = self._diff_score(candidate.image, last_kept.image)
            if score >= self.change_threshold:
                kept.append(candidate)

        if len(ordered) > 1:
            last_frame = ordered[-1]
            if last_frame.frame_index != kept[-1].frame_index:
                kept.append(last_frame)  # always keep the last frame for end-of-clip context

        summary = KeyframeSelectionSummary(
            frames_in=len(ordered),
            frames_kept=len(kept),
            frames_discarded=len(ordered) - len(kept),
            change_threshold=self.change_threshold,
        )
        return kept, summary


# --------------------------------------------------------------------------
# LangGraph node function
# --------------------------------------------------------------------------

def keyframe_selection_node(
    state: PipelineStateWithFrames,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
) -> PipelineStateWithFrames:
    """Insert this node between frame_extraction and quality_assessment:

        graph.add_node("keyframe_selection", keyframe_selection_node)
        graph.add_edge("frame_extraction", "keyframe_selection")
        graph.add_edge("keyframe_selection", "quality_assessment")  # Step 3

    Overwrites state["frames"] with only the kept keyframes, and updates
    state["frames_sampled"] to reflect the smaller count. Every
    downstream step (3-8) is unaffected by this — they just see fewer,
    more meaningful frames to work with.
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    selector = KeyframeSelector(change_threshold=change_threshold)
    kept, summary = selector.select(frames)

    return {
        **state,
        "frames": kept,
        "frames_sampled": len(kept),
        "frames_before_keyframe_selection": summary.frames_in,
        "frames_discarded_as_redundant": summary.frames_discarded,
        "error": None,
    }


# --------------------------------------------------------------------------
# Standalone smoke test — chains Step 1 -> 2 -> 2b
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node

    if len(sys.argv) < 2:
        print("Usage: python step2b_keyframe_selection.py <path_to_video> [change_threshold]")
        sys.exit(1)

    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHANGE_THRESHOLD

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    print(f"Frames after Step 2 sampling: {state['frames_sampled']}")

    state = keyframe_selection_node(state, change_threshold=threshold)
    print(f"Frames after keyframe selection: {state['frames_sampled']}")
    print(f"Discarded as redundant: {state['frames_discarded_as_redundant']} "
          f"(threshold={threshold})")

    print("\nKept frame timestamps:")
    for f in state["frames"]:
        print(f"  idx={f.frame_index:5d}  t={f.timestamp_sec:6.2f}s")
