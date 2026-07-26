"""
Step 3: VLM-Based Quality Assessment
======================================

Purpose
-------
For each sampled frame from Step 2, ask a locally-hosted Vision Language
Model to:
    1. Judge visual quality (0-10 score).
    2. Identify the dominant degradation type.
    3. Recommend the enhancement tool best suited to fix it.

This targets a LOCAL open-source VLM served on your GPU node via Ollama
(the simplest self-hosted route for LLaVA-family models). If you're using
a different serving stack (vLLM, TGI, a raw HF `transformers` pipeline,
LMDeploy, etc.), only `LocalVLMClient._call_backend()` needs to change —
everything else (prompt, parsing, node wiring) stays the same.

Why Ollama by default:
    - `ollama pull llava` gets you a running local endpoint in one command.
    - REST API on http://localhost:11434 (or your GPU node's host:port).
    - Accepts base64-encoded images directly — no separate image server needed.

Setup on the GPU node (one-time):
    ollama pull llava          # or llava:13b / llava:34b / bakllava
    ollama serve                # if not already running as a service

Design notes
------------
- The VLM is asked to respond in strict JSON so this stays a *deterministic*
  pipeline stage rather than free-text the ReAct agent (Step 4) has to
  re-parse. A regex-based fallback parser handles models that wrap JSON in
  prose despite instructions.
- If the VLM call fails (GPU node down, model not pulled, timeout), the
  node does NOT crash the graph — it falls back to the cheap OpenCV
  heuristic flags already computed in Step 1 (`degradation_flags`), tags
  the frame as `vlm_unavailable`, and lets the ReAct agent decide whether
  to proceed, retry, or halt.
- One VLM call per sampled frame. For long videos at higher sampling
  rates, consider batching or a max-frames cap — left as a tunable.
"""

from __future__ import annotations

import base64
import json
import re
import requests
import cv2
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord


# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llava"
REQUEST_TIMEOUT_SEC = 60

# Enhancement tools the VLM is allowed to recommend — keep this in sync
# with the tool registry Step 5 actually implements.
VALID_TOOL_RECOMMENDATIONS = {
    "real_esrgan",       # super resolution
    "deblurgan_v2",      # motion blur
    "restormer",         # denoising / general restoration
    "swinir",             # artifact removal / super resolution
    "histogram_equalization",  # cheap, traditional — poor lighting/contrast
    "sharpen",            # cheap, traditional — mild blur
    "denoise_cv2",         # cheap, traditional — mild noise
    "none",                # frame is already acceptable
}

QUALITY_ASSESSMENT_PROMPT = """You are a video quality inspector for a surveillance enhancement pipeline.
Look at this frame and respond with ONLY a JSON object (no other text), exactly in this shape:

{
  "quality_score": <integer 0-10, 10 = pristine>,
  "degradation_type": "<one of: motion_blur, low_resolution, noise, compression_artifacts, poor_lighting, pixel_loss, none>",
  "recommended_tool": "<one of: real_esrgan, deblurgan_v2, restormer, swinir, histogram_equalization, sharpen, denoise_cv2, none>",
  "needs_enhancement": <true or false>,
  "reasoning": "<one short sentence>"
}

Base the recommendation on severity: prefer cheap traditional tools (sharpen, denoise_cv2, histogram_equalization) for mild issues, and reserve GPU deep-learning tools (real_esrgan, deblurgan_v2, restormer, swinir) for moderate-to-severe degradation."""


@dataclass
class VLMObservation:
    frame_index: int
    timestamp_sec: float
    quality_score: Optional[int]
    degradation_type: Optional[str]
    recommended_tool: Optional[str]
    needs_enhancement: Optional[bool]
    reasoning: Optional[str]
    source: str  # "vlm" or "heuristic_fallback"


# --------------------------------------------------------------------------
# 2. Local VLM client
# --------------------------------------------------------------------------

class LocalVLMClient:
    """Thin client for a self-hosted VLM on the GPU node. Defaults to the
    Ollama REST API; swap `_call_backend` to point at vLLM/TGI/etc."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model

    @staticmethod
    def _encode_frame(image) -> str:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to JPEG-encode frame for VLM request.")
        return base64.b64encode(buf).decode("utf-8")

    def _call_backend(self, image_b64: str, prompt: str) -> str:
        """Ollama-specific call. Returns the raw text response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",  # Ollama will try to constrain to valid JSON
        }
        resp = requests.post(self.base_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def assess_frame(self, image) -> Dict[str, Any]:
        image_b64 = self._encode_frame(image)
        raw_text = self._call_backend(image_b64, QUALITY_ASSESSMENT_PROMPT)
        return self._parse_response(raw_text)

    @staticmethod
    def _parse_response(raw_text: str) -> Dict[str, Any]:
        # Strict attempt first
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        # Fallback: pull the first {...} block out of prose the model may
        # have wrapped around the JSON despite instructions.
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse VLM response as JSON: {raw_text[:200]!r}")


# --------------------------------------------------------------------------
# 3. Fallback: reuse Step 1's cheap heuristics when the VLM is unreachable
# --------------------------------------------------------------------------

_HEURISTIC_TO_TOOL = {
    "motion_blur": "deblurgan_v2",
    "low_resolution": "real_esrgan",
    "poor_lighting": "histogram_equalization",
    "noise_or_compression_artifacts": "restormer",
    "dropped_or_corrupted_frames": "none",  # can't enhance away missing frames
}


def _heuristic_observation(frame: FrameRecord, degradation_flags: List[str]) -> VLMObservation:
    flag = degradation_flags[0] if degradation_flags else "none"
    tool = _HEURISTIC_TO_TOOL.get(flag, "none")
    return VLMObservation(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        quality_score=None,
        degradation_type=flag,
        recommended_tool=tool,
        needs_enhancement=flag != "none",
        reasoning="VLM unavailable — used Step 1 heuristic flags as fallback.",
        source="heuristic_fallback",
    )


# --------------------------------------------------------------------------
# 4. LangGraph node function
# --------------------------------------------------------------------------

def quality_assessment_node(
    state: PipelineStateWithFrames,
    vlm_client: Optional[LocalVLMClient] = None,
    max_frames: Optional[int] = None,
) -> PipelineStateWithFrames:
    """Third node in the graph. Wire it in as:

        graph.add_node("quality_assessment", quality_assessment_node)
        graph.add_edge("frame_extraction", "quality_assessment")
        graph.add_edge("quality_assessment", "react_agent")   # Step 4

    Populates state["vlm_observations"] — one entry per sampled frame,
    plus per-frame .quality_notes on each FrameRecord for convenience.
    """
    if state.get("error"):
        return state  # upstream failure — don't waste GPU calls

    frames: List[FrameRecord] = state.get("frames", [])
    degradation_flags = state.get("degradation_flags", [])
    client = vlm_client or LocalVLMClient()

    frames_to_process = frames[:max_frames] if max_frames else frames
    observations: List[VLMObservation] = []
    vlm_failures = 0

    for frame in frames_to_process:
        try:
            result = client.assess_frame(frame.image)
            obs = VLMObservation(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                quality_score=result.get("quality_score"),
                degradation_type=result.get("degradation_type"),
                recommended_tool=result.get("recommended_tool")
                if result.get("recommended_tool") in VALID_TOOL_RECOMMENDATIONS else "none",
                needs_enhancement=result.get("needs_enhancement"),
                reasoning=result.get("reasoning"),
                source="vlm",
            )
        except (requests.RequestException, ValueError, TimeoutError) as exc:
            vlm_failures += 1
            obs = _heuristic_observation(frame, degradation_flags)
            obs.reasoning = f"VLM call failed ({exc}); used heuristic fallback."

        frame.quality_notes = f"{obs.degradation_type} -> {obs.recommended_tool} (score={obs.quality_score})"
        observations.append(obs)

    return {
        **state,
        "frames": frames,
        "vlm_observations": [asdict(o) for o in observations],
        "vlm_failures": vlm_failures,
        "error": None,
    }


# --------------------------------------------------------------------------
# 5. Standalone smoke test — chains Step 1 -> 2 -> 3
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node

    if len(sys.argv) < 2:
        print("Usage: python step3_quality_assessment.py <path_to_video> [ollama_url] [model]")
        sys.exit(1)

    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OLLAMA_URL
    model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    print(f"Sampled {state['frames_sampled']} frames — sending to VLM at {url} (model={model})...")

    client = LocalVLMClient(base_url=url, model=model)
    state = quality_assessment_node(state, vlm_client=client)

    print(f"VLM failures: {state.get('vlm_failures', 0)} / {len(state['vlm_observations'])}")
    for obs in state["vlm_observations"]:
        print(
            f"  t={obs['timestamp_sec']:6.2f}s  score={obs['quality_score']}  "
            f"degradation={obs['degradation_type']:25s}  tool={obs['recommended_tool']:20s}  "
            f"[{obs['source']}]"
        )
