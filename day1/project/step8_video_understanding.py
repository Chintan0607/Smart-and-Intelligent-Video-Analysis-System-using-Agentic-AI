"""
Step 8: Video Understanding
==============================

Purpose
-------
Final stage of the pipeline. Takes the reconstructed/enhanced frames and
produces the natural-language deliverable described in the project spec:
an event summary, object/activity descriptions, a timeline, and (where
relevant) a note on anything that looks unusual — in the style of:

    "At 02:15 PM, a person entered through the main gate carrying a
    backpack..."

Approach (two-pass, since llava is a per-image VLM, not a native video
model):

1. CAPTION PASS — send each accepted frame (with its timestamp) to the
   VLM individually, asking for a factual, concrete description of what's
   visible: objects, people, actions, notable details. This produces a
   per-timestamp caption list.

2. SYNTHESIS PASS — feed the full list of timestamped captions to the
   same model (text-only, no image) and ask it to write ONE coherent
   report: a summary, a timeline of what happened across the clip, and
   an explicit call-out of anything that looks suspicious or worth a
   human reviewer's attention.

This mirrors how you'd realistically get video-level understanding out of
an image-only VLM — caption the frames, then have the LLM stitch the
captions into a narrative — rather than pretending llava can reason over
the whole video in one call (it can't; Ollama's image support is
per-request, single or multi-image, not temporal-video-aware).

Reliability
-----------
- Caption failures are non-fatal per frame — a frame that fails to
  caption is noted as "[caption unavailable]" in the timeline rather than
  crashing the whole report.
- If ALL captions fail (e.g. VLM totally unreachable), the report step is
  skipped and state["error"] is set — no point synthesizing a report from
  nothing.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord
from step3_quality_assessment import LocalVLMClient, DEFAULT_OLLAMA_URL, DEFAULT_MODEL

REQUEST_TIMEOUT_SEC = 60

CAPTION_PROMPT = """You are analyzing a frame from surveillance footage. Describe factually and
concretely what is visible in this frame: people, objects, vehicles, animals, actions being taken,
and their approximate position/movement if apparent. Be specific and objective — do not speculate
beyond what is visibly shown. Respond in 1-3 plain sentences, no JSON, no preamble."""

SUMMARY_PROMPT_TEMPLATE = """Below are frame-by-frame descriptions with timestamps, from a single
continuous surveillance video clip.

{timestamped_captions}

Write a 2-3 sentence SUMMARY of what this clip shows overall. Respond with ONLY the summary text,
no headers, no preamble."""

TIMELINE_PROMPT_TEMPLATE = """Below are frame-by-frame descriptions with timestamps, from a single
continuous surveillance video clip.

{timestamped_captions}

Write a chronological TIMELINE of events as flowing prose, referencing timestamps, describing how
the scene changes/progresses across the clip. Do not just repeat each caption verbatim — synthesize
what is actually happening over time. Respond with ONLY the timeline text, no headers, no preamble."""

NOTABLE_PROMPT_TEMPLATE = """Below are frame-by-frame descriptions with timestamps, from a single
continuous surveillance video clip.

{timestamped_captions}

Identify anything unusual or worth a human reviewer's attention (unfamiliar people, unattended
objects, unusual movement, etc). If nothing stands out, respond with exactly: "Nothing suspicious
observed." Otherwise respond with 1-3 sentences describing what's notable. Respond with ONLY that
text, no headers, no preamble."""


@dataclass
class FrameCaption:
    frame_index: int
    timestamp_sec: float
    caption: str
    success: bool


class VideoUnderstandingClient:
    """Reuses the same Ollama-served llava model — one call per frame for
    captioning, one final call (text-only) for narrative synthesis."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self._vlm = LocalVLMClient(base_url=base_url, model=model)

    def caption_frame(self, image) -> str:
        image_b64 = self._vlm._encode_frame(image)
        payload = {
            "model": self.model,
            "prompt": CAPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
        }
        resp = requests.post(self.base_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _call_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(self.base_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def synthesize_report(self, captions: List[FrameCaption]) -> str:
        lines = [
            f"- t={c.timestamp_sec:.2f}s: {c.caption}"
            for c in captions if c.success
        ]
        caption_block = "\n".join(lines)

        summary = self._call_text(SUMMARY_PROMPT_TEMPLATE.format(timestamped_captions=caption_block))
        timeline = self._call_text(TIMELINE_PROMPT_TEMPLATE.format(timestamped_captions=caption_block))
        notable = self._call_text(NOTABLE_PROMPT_TEMPLATE.format(timestamped_captions=caption_block))

        return (
            f"SUMMARY\n{summary}\n\n"
            f"TIMELINE\n{timeline}\n\n"
            f"NOTABLE / SUSPICIOUS ACTIVITY\n{notable}"
        )


# --------------------------------------------------------------------------
# LangGraph node function
# --------------------------------------------------------------------------

def video_understanding_node(
    state: PipelineStateWithFrames,
    client: Optional[VideoUnderstandingClient] = None,
) -> PipelineStateWithFrames:
    """Eighth and final node in the graph. Wire it in as:

        graph.add_node("video_understanding", video_understanding_node)
        graph.add_edge("reconstruction", "video_understanding")
        graph.set_finish_point("video_understanding")

    Writes state["frame_captions"] (per-frame descriptions) and
    state["final_report"] (the synthesized narrative report).
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    if not frames:
        return {**state, "error": "video_understanding failed: no frames in state."}

    vu_client = client or VideoUnderstandingClient()
    ordered = sorted(frames, key=lambda f: f.frame_index)

    captions: List[FrameCaption] = []
    for frame in ordered:
        try:
            text = vu_client.caption_frame(frame.image)
            captions.append(FrameCaption(frame.frame_index, frame.timestamp_sec, text, True))
        except (requests.RequestException, ValueError, TimeoutError) as exc:
            captions.append(FrameCaption(
                frame.frame_index, frame.timestamp_sec,
                f"[caption unavailable: {exc}]", False,
            ))

    successful = [c for c in captions if c.success]
    if not successful:
        return {
            **state,
            "frame_captions": [asdict(c) for c in captions],
            "error": "video_understanding failed: no frame captions succeeded — VLM may be unreachable.",
        }

    try:
        report = vu_client.synthesize_report(captions)
    except (requests.RequestException, ValueError, TimeoutError) as exc:
        return {
            **state,
            "frame_captions": [asdict(c) for c in captions],
            "error": f"video_understanding failed during report synthesis: {exc}",
        }

    return {
        **state,
        "frame_captions": [asdict(c) for c in captions],
        "final_report": report,
        "error": None,
    }


# --------------------------------------------------------------------------
# Standalone smoke test — chains the FULL pipeline, Step 1 through Step 8
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step3_quality_assessment import quality_assessment_node
    from step4_react_agent import react_agent_node, ReActAgent, MAX_ENHANCEMENT_ROUNDS
    from step5_tool_execution import tool_execution_node
    from step6_validation_loop import validation_node, route_after_validation
    from step7_reconstruction import reconstruction_node

    if len(sys.argv) < 2:
        print("Usage: python step8_video_understanding.py <path_to_video>")
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

    print("Reconstructing enhanced video...")
    state = reconstruction_node(state)
    if state.get("error"):
        print("Reconstruction failed:", state["error"])
        sys.exit(1)
    print(f"Enhanced video written to: {state['reconstructed_video_path']}")

    print("\nGenerating video understanding report (captioning frames)...")
    state = video_understanding_node(state, client=VideoUnderstandingClient())
    if state.get("error"):
        print("Video understanding failed:", state["error"])
        sys.exit(1)

    print("\nPer-frame captions:")
    for c in state["frame_captions"]:
        print(f"  t={c['timestamp_sec']:6.2f}s: {c['caption']}")

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(state["final_report"])
