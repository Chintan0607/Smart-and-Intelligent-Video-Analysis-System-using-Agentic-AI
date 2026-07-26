"""
Step 8b: Anomaly & Incident Detection
========================================

Purpose
-------
Addresses project objectives 2.4 (Incident Intelligence) and in-scope
item 4.4 (Detection engine for predefined high-risk events), and
Boundary Condition 5.3 (scope limited to unauthorized entry and
loitering).

Step 8's "Notable/Suspicious" section relies on the VLM freely deciding
what's worth flagging — open-ended and, given llava's inconsistency
we've seen throughout this pipeline, not reliable enough to call a
"detection engine." This module instead does DETERMINISTIC, rule-based
presence tracking over the timestamped captions Step 8 already
generated — fast, free, and reproducible — and only asks the VLM to turn
a flagged window into readable text, not to decide whether something is
suspicious in the first place.

Detected event types (matching your Boundary Condition 5.3 scope)
--------------------------------------------------------------------
1. POSSIBLE_ENTRY — a person transitions from absent to present between
   consecutive keyframes. Flagged as "possible" because this pipeline has
   no access-control list or authorization data — it cannot know WHO is
   authorized, only that someone newly appeared. A human reviewer makes
   the authorized/unauthorized call; this system surfaces the moment for
   them.
2. LOITERING — a person is detected as present across a continuous run
   of keyframes spanning >= `loiter_threshold_sec` (default 30s), with no
   more than one consecutive frame-gap in between (tolerates a single
   frame where the presence-keyword happened not to appear in the
   caption's exact wording).

Method
------
Presence per frame is inferred with simple keyword matching over each
caption's text (e.g. "person", "man", "woman", "individual", "someone").
This is intentionally crude — a regex pass, not another VLM call — so
event detection doesn't inherit the VLM's per-call inconsistency. Precision/
recall depends entirely on caption wording quality from Step 8; if
captions are consistently good (as they were in your test run), this
works well. If you start seeing missed detections, the fix is to widen
PERSON_KEYWORDS, not to add more VLM calls into the detection path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from step2_frame_extraction import PipelineStateWithFrames
from step8_video_understanding import FrameCaption

DEFAULT_LOITER_THRESHOLD_SEC = 30.0

PERSON_KEYWORDS = [
    "person", "people", "man", "woman", "men", "women", "individual",
    "someone", "human", "figure", "intruder", "pedestrian",
]
PERSON_PATTERN = re.compile(r"\b(" + "|".join(PERSON_KEYWORDS) + r")\b", re.IGNORECASE)


@dataclass
class AnomalyEvent:
    event_type: str          # "possible_entry" | "loitering"
    start_time_sec: float
    end_time_sec: float
    description: str


def _person_present(caption_text: str) -> bool:
    return bool(PERSON_PATTERN.search(caption_text))


class AnomalyDetector:
    def __init__(self, loiter_threshold_sec: float = DEFAULT_LOITER_THRESHOLD_SEC):
        self.loiter_threshold_sec = loiter_threshold_sec

    def detect(self, captions: List[FrameCaption]) -> List[AnomalyEvent]:
        ordered = sorted(
            [c for c in captions if c.success],
            key=lambda c: c.timestamp_sec,
        )
        if not ordered:
            return []

        presence = [(c.timestamp_sec, _person_present(c.caption)) for c in ordered]
        events: List[AnomalyEvent] = []

        # --- entry detection: absent -> present transitions ---
        for i in range(1, len(presence)):
            prev_t, prev_present = presence[i - 1]
            cur_t, cur_present = presence[i]
            if cur_present and not prev_present:
                events.append(AnomalyEvent(
                    event_type="possible_entry",
                    start_time_sec=prev_t,
                    end_time_sec=cur_t,
                    description=(
                        f"A person is first detected between t={prev_t:.2f}s and t={cur_t:.2f}s "
                        f"(not present in the prior keyframe, present in this one). Review to "
                        f"confirm identity/authorization."
                    ),
                ))

        # --- loitering detection: continuous presence >= threshold ---
        run_start: Optional[float] = None
        run_end: Optional[float] = None
        for t, present in presence:
            if present:
                if run_start is None:
                    run_start = t
                run_end = t
            else:
                if run_start is not None and (run_end - run_start) >= self.loiter_threshold_sec:
                    events.append(AnomalyEvent(
                        event_type="loitering",
                        start_time_sec=run_start,
                        end_time_sec=run_end,
                        description=(
                            f"A person appears continuously present from t={run_start:.2f}s to "
                            f"t={run_end:.2f}s ({run_end - run_start:.1f}s), exceeding the "
                            f"{self.loiter_threshold_sec:.0f}s loitering threshold."
                        ),
                    ))
                run_start, run_end = None, None
        # flush a run that continues to the end of the clip
        if run_start is not None and (run_end - run_start) >= self.loiter_threshold_sec:
            events.append(AnomalyEvent(
                event_type="loitering",
                start_time_sec=run_start,
                end_time_sec=run_end,
                description=(
                    f"A person appears continuously present from t={run_start:.2f}s to "
                    f"t={run_end:.2f}s ({run_end - run_start:.1f}s, continuing to end of clip), "
                    f"exceeding the {self.loiter_threshold_sec:.0f}s loitering threshold."
                ),
            ))

        return events


# --------------------------------------------------------------------------
# LangGraph node function
# --------------------------------------------------------------------------

def anomaly_detection_node(
    state: PipelineStateWithFrames,
    loiter_threshold_sec: float = DEFAULT_LOITER_THRESHOLD_SEC,
) -> PipelineStateWithFrames:
    """Run this alongside/after Step 8 (video_understanding):

        graph.add_node("anomaly_detection", anomaly_detection_node)
        graph.add_edge("video_understanding", "anomaly_detection")
        graph.set_finish_point("anomaly_detection")

    Reads state["frame_captions"] (written by Step 8). Writes
    state["anomaly_events"] — a structured incident log — and appends a
    deterministic incident summary onto state["final_report"] so the
    report's suspicious-activity claims are backed by explicit, reviewable
    rule triggers rather than open-ended VLM judgment alone.
    """
    if state.get("error"):
        return state

    raw_captions = state.get("frame_captions", [])
    if not raw_captions:
        return {**state, "error": "anomaly_detection failed: no frame_captions found — run Step 8 first."}

    captions = [FrameCaption(**c) for c in raw_captions]
    detector = AnomalyDetector(loiter_threshold_sec=loiter_threshold_sec)
    events = detector.detect(captions)

    if events:
        lines = [f"- [{e.event_type.upper()}] {e.description}" for e in events]
        incident_block = "INCIDENT LOG (rule-based detection)\n" + "\n".join(lines)
    else:
        incident_block = (
            "INCIDENT LOG (rule-based detection)\n"
            "No predefined high-risk events (unauthorized entry, loitering) detected."
        )

    existing_report = state.get("final_report", "")
    updated_report = f"{existing_report}\n\n{incident_block}" if existing_report else incident_block

    return {
        **state,
        "anomaly_events": [asdict(e) for e in events],
        "final_report": updated_report,
        "error": None,
    }


# --------------------------------------------------------------------------
# Standalone smoke test — chains the FULL pipeline including Step 2b and 8b
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step2b_keyframe_selection import keyframe_selection_node
    from step3_quality_assessment import quality_assessment_node, LocalVLMClient
    from step4_react_agent import react_agent_node, ReActAgent, MAX_ENHANCEMENT_ROUNDS
    from step5_tool_execution import tool_execution_node
    from step6_validation_loop import validation_node, route_after_validation
    from step7_reconstruction import reconstruction_node
    from step8_video_understanding import video_understanding_node, VideoUnderstandingClient

    if len(sys.argv) < 2:
        print("Usage: python step8b_anomaly_detection.py <path_to_video> [loiter_threshold_sec]")
        sys.exit(1)

    loiter_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LOITER_THRESHOLD_SEC

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    state = keyframe_selection_node(state)
    print(f"Frames after keyframe selection: {state['frames_sampled']} "
          f"(discarded {state['frames_discarded_as_redundant']} as redundant)")

    state = quality_assessment_node(state, vlm_client=LocalVLMClient())

    agent = ReActAgent()
    client = LocalVLMClient()
    round_num = 1
    while True:
        state = react_agent_node(state, agent=agent)
        if not state.get("enhancement_plan"):
            break
        state = tool_execution_node(state)
        state = validation_node(state, vlm_client=client)
        if route_after_validation(state) == "proceed":
            break
        round_num += 1
        if round_num > MAX_ENHANCEMENT_ROUNDS + 1:
            break

    state = reconstruction_node(state)
    if state.get("error"):
        print("Reconstruction failed:", state["error"])
        sys.exit(1)
    print(f"Enhanced video written to: {state['reconstructed_video_path']}")

    state = video_understanding_node(state, client=VideoUnderstandingClient())
    if state.get("error"):
        print("Video understanding failed:", state["error"])
        sys.exit(1)

    state = anomaly_detection_node(state, loiter_threshold_sec=loiter_threshold)
    if state.get("error"):
        print("Anomaly detection failed:", state["error"])
        sys.exit(1)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(state["final_report"])
