"""
Step 6: Quality Validation Loop
==================================

Purpose
-------
After Step 5 applies a tool, send the ENHANCED frame back to the VLM,
get a fresh quality_score, and compare it objectively against the
ORIGINAL score that Step 3 recorded before any enhancement:

    improved = new_score >= original_score

- If improved (or the frame's action was "none" — agent chose not to
  touch it) -> mark the frame ACCEPTED. It's done; Steps 4/5 will skip
  it on any future loop (see the accepted_frames skip added in Step 4).
- If not improved and rounds remain -> leave it PENDING. The graph loops
  back to Step 4, which will see the failed tool in enhancement_history
  and try something else.
- If not improved and MAX_ENHANCEMENT_ROUNDS is hit -> mark ACCEPTED
  anyway (forced), so the frame doesn't loop forever. It moves on with
  whatever quality it ended up at — Step 8's report can still mention it
  came from a frame that didn't fully recover.

This node also refreshes `vlm_observations` for retried frames with the
new score/degradation/reasoning, so Step 4's next round reasons from
current data instead of the pre-enhancement observation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord
from step3_quality_assessment import LocalVLMClient
from step4_react_agent import MAX_ENHANCEMENT_ROUNDS

import requests


@dataclass
class ValidationResult:
    frame_index: int
    original_score: Optional[int]
    new_score: Optional[int]
    improved: bool
    accepted: bool
    reason: str


# --------------------------------------------------------------------------
# LangGraph node function
# --------------------------------------------------------------------------

def validation_node(
    state: PipelineStateWithFrames,
    vlm_client: Optional[LocalVLMClient] = None,
) -> PipelineStateWithFrames:
    """Sixth node in the graph. Wire it in as:

        graph.add_node("validation", validation_node)
        graph.add_edge("tool_execution", "validation")
        graph.add_conditional_edges(
            "validation",
            route_after_validation,
            {"retry": "react_agent", "proceed": "reconstruction"},   # Step 7
        )

    Writes state["accepted_frames"] (list of frame_index that are done)
    and state["validation_results"] (per-frame score comparison, for
    logging/debugging/the final report).
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    plan_by_index = {d["frame_index"]: d for d in state.get("enhancement_plan", [])}
    observations = state.get("vlm_observations", [])
    obs_by_index = {o["frame_index"]: o for o in observations}
    accepted = set(state.get("accepted_frames", []))
    client = vlm_client or LocalVLMClient()

    results: List[ValidationResult] = []
    updated_observations = list(observations)

    for frame in frames:
        idx = frame.frame_index
        if idx in accepted:
            continue  # already done, nothing to validate again

        decision = plan_by_index.get(idx)
        original_obs = obs_by_index.get(idx, {})
        original_score = original_obs.get("quality_score")

        if decision is None:
            # No decision was made for this frame this round (shouldn't
            # normally happen, but guard against it) — accept as-is.
            accepted.add(idx)
            results.append(ValidationResult(
                idx, original_score, original_score, False, True,
                "No enhancement decision found this round — accepted as-is.",
            ))
            continue

        action = decision.get("action", "none")
        round_num = len(frame.enhancement_history)

        if action == "none":
            # Agent deliberately chose not to enhance — respect that and accept.
            accepted.add(idx)
            results.append(ValidationResult(
                idx, original_score, original_score, False, True,
                "Agent chose no action — accepted as-is.",
            ))
            continue

        try:
            fresh = client.assess_frame(frame.image)
            new_score = fresh.get("quality_score")
        except (requests.RequestException, ValueError, TimeoutError) as exc:
            # Can't validate right now — leave pending for retry rather
            # than falsely accepting or failing the whole pipeline.
            results.append(ValidationResult(
                idx, original_score, None, False, False,
                f"VLM re-check failed ({exc}) — left pending for retry.",
            ))
            continue

        strictly_improved = (
            new_score is not None and original_score is not None
            and new_score > original_score
        )
        unchanged = (
            new_score is not None and original_score is not None
            and new_score == original_score
        )
        improved = strictly_improved or unchanged  # tie or better = accept, no regression
        maxed_out = round_num >= MAX_ENHANCEMENT_ROUNDS
        is_accepted = improved or maxed_out

        if is_accepted:
            accepted.add(idx)

        if strictly_improved:
            reason = f"Improved: {original_score} -> {new_score}."
        elif unchanged:
            reason = f"No change ({original_score} -> {new_score}) but no regression — accepted."
        elif maxed_out:
            reason = f"No improvement ({original_score} -> {new_score}) but max rounds reached — accepted as final."
        else:
            reason = f"No improvement ({original_score} -> {new_score}) — will retry with a different tool."

        results.append(ValidationResult(idx, original_score, new_score, improved, is_accepted, reason))

        # Refresh this frame's observation with the latest read, so the
        # next react_agent round (if any) reasons from current data.
        for i, o in enumerate(updated_observations):
            if o.get("frame_index") == idx:
                updated_observations[i] = {
                    **o,
                    "quality_score": new_score,
                    "degradation_type": fresh.get("degradation_type", o.get("degradation_type")),
                    "reasoning": fresh.get("reasoning", o.get("reasoning")),
                }
                break

    return {
        **state,
        "accepted_frames": sorted(accepted),
        "vlm_observations": updated_observations,
        "validation_results": [asdict(r) for r in results],
        "error": None,
    }


def route_after_validation(state: PipelineStateWithFrames) -> str:
    """Conditional edge function for LangGraph. Returns 'retry' if any
    frame still needs another enhancement round, else 'proceed'."""
    frames: List[FrameRecord] = state.get("frames", [])
    accepted = set(state.get("accepted_frames", []))
    pending = [f for f in frames if f.frame_index not in accepted]
    return "retry" if pending else "proceed"


# --------------------------------------------------------------------------
# Standalone smoke test — chains Steps 1 -> 6 with the retry loop run
# manually (no LangGraph wiring needed to see the loop behavior)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step3_quality_assessment import quality_assessment_node
    from step4_react_agent import react_agent_node, ReActAgent
    from step5_tool_execution import tool_execution_node

    if len(sys.argv) < 2:
        print("Usage: python step6_validation_loop.py <path_to_video>")
        sys.exit(1)

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    state = quality_assessment_node(state, vlm_client=LocalVLMClient())
    print(f"Sampled {state['frames_sampled']} frames, initial VLM pass done.\n")

    agent = ReActAgent()
    client = LocalVLMClient()
    round_num = 1

    while True:
        print(f"=== Round {round_num} ===")
        state = react_agent_node(state, agent=agent)
        if not state.get("enhancement_plan"):
            print("No pending decisions — all frames already accepted.")
            break

        state = tool_execution_node(state)
        state = validation_node(state, vlm_client=client)

        for r in state["validation_results"]:
            print(f"  frame {r['frame_index']:4d}  {r['original_score']} -> {r['new_score']}  "
                  f"accepted={r['accepted']}  {r['reason']}")

        decision = route_after_validation(state)
        print(f"  -> route: {decision}\n")
        if decision == "proceed":
            break
        round_num += 1
        if round_num > MAX_ENHANCEMENT_ROUNDS + 1:
            print("Safety stop: exceeded expected round count.")
            break

    print("Final accepted frames:", state.get("accepted_frames"))