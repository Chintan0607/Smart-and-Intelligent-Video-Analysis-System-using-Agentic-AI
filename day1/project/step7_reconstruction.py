"""
Step 7: Frame Reconstruction
===============================

Purpose
-------
Takes the final set of frames (post Step 6 validation — every frame is
either genuinely improved, unchanged-but-accepted, or force-accepted
after hitting MAX_ENHANCEMENT_ROUNDS) and reassembles them into a single
output video file.

Two decisions baked in per your calls:

1. RESOLUTION MISMATCH: some frames (e.g. from the real_esrgan stub) may
   have been upscaled and are now a different resolution than the rest.
   This node finds the LARGEST width/height among all frames and resizes
   every other frame UP to match it (cv2.INTER_CUBIC), rather than
   downscaling the enhanced ones back down and throwing away the detail
   gain they got.

2. FRAME RATE: output is written at the SAMPLED rate (state["sampling_fps"],
   ~1fps by default from Step 2) — not the original source fps. This
   produces a short, choppy preview of just the frames that were actually
   assessed/enhanced, rather than stretching them across a smooth 30fps
   timeline where 29 out of every 30 frames would be duplicates. This is
   the right call for validating the pipeline; if you later want a smooth
   full-length output video, that requires either extracting/enhancing
   every frame in Step 2, or interpolating between sampled frames — both
   bigger changes than Step 7 alone should make.

Order preservation: frames are always written in ascending frame_index
order, matching their original position in the source video, regardless
of what order they were processed/accepted in during the validation loop.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from typing import List, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord

DEFAULT_OUTPUT_DIR = "data/enhanced"
FOURCC = "mp4v"


def _target_resolution(frames: List[FrameRecord]) -> tuple[int, int]:
    """Returns (width, height) — the largest of each dimension found
    across all frames (not necessarily from the same frame)."""
    max_w = max(f.image.shape[1] for f in frames)
    max_h = max(f.image.shape[0] for f in frames)
    return max_w, max_h


def _resize_to(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if (w, h) == (target_w, target_h):
        return image
    # Upscaling benefits from CUBIC; only downscales (shouldn't happen given
    # we always resize to the max) would prefer AREA — CUBIC is fine either way.
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def reconstruct_video(
    frames: List[FrameRecord],
    output_path: str,
    fps: float,
) -> str:
    if not frames:
        raise ValueError("No frames to reconstruct — cannot write an empty video.")

    ordered = sorted(frames, key=lambda f: f.frame_index)
    target_w, target_h = _target_resolution(ordered)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for '{output_path}'. Check codec/output path.")

    try:
        for frame in ordered:
            resized = _resize_to(frame.image, target_w, target_h)
            writer.write(resized)
    finally:
        writer.release()

    return output_path


# --------------------------------------------------------------------------
# LangGraph node function
# --------------------------------------------------------------------------

def reconstruction_node(
    state: PipelineStateWithFrames,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> PipelineStateWithFrames:
    """Seventh node in the graph. Wire it in as:

        graph.add_node("reconstruction", reconstruction_node)
        graph.add_edge("validation", "reconstruction")  # only on the "proceed" branch
        graph.add_edge("reconstruction", "video_understanding")  # Step 8

    Requires all frames to be in state["accepted_frames"] (Step 6's job to
    guarantee before routing here). Writes the reassembled video to
    <output_dir>/<video_id>_enhanced.mp4 and records the path in
    state["reconstructed_video_path"].
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    if not frames:
        return {**state, "error": "reconstruction failed: no frames in state."}

    video_id = state.get("video_id", "output")
    fps = state.get("sampling_fps", 1.0)
    output_path = os.path.join(output_dir, f"{video_id}_enhanced.mp4")

    try:
        final_path = reconstruct_video(frames, output_path, fps)
    except (ValueError, RuntimeError) as exc:
        return {**state, "error": f"reconstruction failed: {exc}"}

    return {
        **state,
        "reconstructed_video_path": final_path,
        "error": None,
    }


# --------------------------------------------------------------------------
# Standalone smoke test — chains the full pipeline through reconstruction
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step3_quality_assessment import quality_assessment_node, LocalVLMClient
    from step4_react_agent import react_agent_node, ReActAgent, MAX_ENHANCEMENT_ROUNDS
    from step5_tool_execution import tool_execution_node
    from step6_validation_loop import validation_node, route_after_validation

    if len(sys.argv) < 2:
        print("Usage: python step7_reconstruction.py <path_to_video>")
        sys.exit(1)

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    state = quality_assessment_node(state, vlm_client=LocalVLMClient())
    print(f"Sampled {state['frames_sampled']} frames, initial VLM pass done.")

    agent = ReActAgent()
    client = LocalVLMClient()
    round_num = 1
    while True:
        state = react_agent_node(state, agent=agent)
        if not state.get("enhancement_plan"):
            break
        state = tool_execution_node(state)
        state = validation_node(state, vlm_client=client)
        print(f"Round {round_num}: accepted so far = {len(state.get('accepted_frames', []))}/{state['frames_sampled']}")
        if route_after_validation(state) == "proceed":
            break
        round_num += 1
        if round_num > MAX_ENHANCEMENT_ROUNDS + 1:
            break

    print("\nReconstructing enhanced video...")
    state = reconstruction_node(state)
    if state.get("error"):
        print("Reconstruction failed:", state["error"])
        sys.exit(1)

    print(f"Enhanced video written to: {state['reconstructed_video_path']}")