"""
Step 4: ReAct Agent Decision Layer
=====================================

Purpose
-------
This is the decision-making brain of the pipeline. Given Step 3's VLM
observation for a frame (degradation_type, quality_score, prior VLM
reasoning), the agent free-reasons through a Thought -> Action ->
(Observation) loop and DECIDES which tool to apply next — rather than
being forced through a fixed degradation->tool lookup table.

Per your call: the agent is NOT constrained to Step 3's `recommended_tool`
or to a hardcoded mapping. It reasons from the raw observation each time
and picks freely from the tool registry below. This is more "agentic" and
can occasionally pick something suboptimal — that's an accepted tradeoff,
and it's exactly why Step 6 closes the loop by re-checking quality after
the chosen tool runs, rather than trusting the first decision blindly.

Model
-----
Reasoning is powered by the same local `llava` model already pulled for
Step 3 (via Ollama's /api/generate, text-only this time — no image
payload needed since the agent reasons over Step 3's *description* of
the frame, not the pixels themselves). LLaVA wasn't built for pure text
reasoning/tool-use, so responses are guided with a strict ReAct-style
prompt template and parsed defensively; if parsing fails or the model
names a tool outside the registry, the node falls back to "no_action"
for that frame rather than crashing the graph, and logs it in the trace.

Loop safety
-----------
Step 6 will route frames that still fail quality checks back through this
node. `enhancement_history` on each observation is used to (a) tell the
agent what's already been tried, so it doesn't repeat a failed tool, and
(b) enforce `MAX_ENHANCEMENT_ROUNDS` so a stubborn frame can't loop forever.
"""

from __future__ import annotations

import json
import re
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord
from step3_quality_assessment import DEFAULT_OLLAMA_URL, DEFAULT_MODEL, VALID_TOOL_RECOMMENDATIONS

REQUEST_TIMEOUT_SEC = 60
MAX_ENHANCEMENT_ROUNDS = 3  # per-frame cap to prevent infinite retry loops

# --------------------------------------------------------------------------
# 1. Tool registry the agent can freely choose from
# --------------------------------------------------------------------------
# Descriptions are what the agent sees in its prompt — keep them accurate,
# since the agent's choice quality depends entirely on these.

TOOL_REGISTRY: Dict[str, str] = {
    "real_esrgan": "Deep-learning super resolution. Best for low resolution or heavily upscaled/blocky footage.",
    "deblurgan_v2": "Deep-learning deblurring. Best for motion blur or camera-shake blur.",
    "restormer": "Deep-learning general restoration. Best for heavy noise, compression artifacts, or mixed degradation.",
    "swinir": "Deep-learning restoration/super-resolution. Alternative to real_esrgan/restormer for complex artifacts.",
    "histogram_equalization": "Cheap traditional contrast fix. Best for poor lighting / low contrast.",
    "sharpen": "Cheap traditional sharpening filter. Best for mild blur only.",
    "denoise_cv2": "Cheap traditional OpenCV denoising. Best for mild noise only.",
    "none": "No enhancement needed — frame quality is already acceptable.",
}

REACT_PROMPT_TEMPLATE = """You are the decision-making agent in a surveillance video enhancement pipeline.
You reason step by step (ReAct style: Thought, then Action) and choose ONE tool to apply to a video frame.

Available tools:
{tool_list}

Frame observation from the vision model:
- degradation_type: {degradation_type}
- quality_score (0-10, 10=best): {quality_score}
- VLM's own reasoning: {vlm_reasoning}
- Tools already tried on this frame this session (avoid repeating a tool that already failed): {history}
- Enhancement round: {round_num} of {max_rounds}

Think about what's actually wrong with this frame and what would genuinely fix it — you are not
required to follow any fixed rule, use your own judgment. Then respond with ONLY a JSON object,
no other text, in exactly this shape:

{{
  "thought": "<your reasoning in 1-2 sentences>",
  "action": "<exactly one tool name from the list above>",
  "expects_improvement": <true or false — do you think this will meaningfully improve the frame>
}}"""


@dataclass
class AgentDecision:
    frame_index: int
    timestamp_sec: float
    round_num: int
    thought: str
    action: str
    expects_improvement: bool
    valid: bool  # False if the model named a tool outside the registry / parsing failed


# --------------------------------------------------------------------------
# 2. ReAct agent
# --------------------------------------------------------------------------

class ReActAgent:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.tool_list_str = "\n".join(f"- {name}: {desc}" for name, desc in TOOL_REGISTRY.items())

    def _call_backend(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        resp = requests.post(self.base_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json().get("response", "")

    @staticmethod
    def _parse(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse agent response as JSON: {raw_text[:200]!r}")

    def decide(
        self,
        degradation_type: str,
        quality_score: Optional[int],
        vlm_reasoning: Optional[str],
        history: List[str],
        round_num: int,
    ) -> Dict[str, Any]:
        prompt = REACT_PROMPT_TEMPLATE.format(
            tool_list=self.tool_list_str,
            degradation_type=degradation_type or "unknown",
            quality_score=quality_score if quality_score is not None else "unknown",
            vlm_reasoning=vlm_reasoning or "none given",
            history=", ".join(history) if history else "none",
            round_num=round_num,
            max_rounds=MAX_ENHANCEMENT_ROUNDS,
        )
        raw = self._call_backend(prompt)
        return self._parse(raw)


# --------------------------------------------------------------------------
# 3. LangGraph node function
# --------------------------------------------------------------------------

def react_agent_node(
    state: PipelineStateWithFrames,
    agent: Optional[ReActAgent] = None,
) -> PipelineStateWithFrames:
    """Fourth node in the graph. Wire it in as:

        graph.add_node("react_agent", react_agent_node)
        graph.add_edge("quality_assessment", "react_agent")
        graph.add_edge("react_agent", "tool_execution")   # Step 5

    Reads state["vlm_observations"] (from Step 3) and state["frames"]
    (for enhancement_history on each FrameRecord). Writes
    state["enhancement_plan"] — one decision per frame, in the same order
    as frames — for Step 5 to execute.

    On the loop-back path from Step 6, the same node is re-entered; it
    reads updated observations and each frame's growing
    enhancement_history to avoid repeating a tool that already failed,
    and respects MAX_ENHANCEMENT_ROUNDS as a circuit breaker.
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    observations = state.get("vlm_observations", [])
    accepted_frames = set(state.get("accepted_frames", []))
    react_agent = agent or ReActAgent()

    obs_by_index = {o["frame_index"]: o for o in observations}
    decisions: List[AgentDecision] = []

    for frame in frames:
        if frame.frame_index in accepted_frames:
            continue  # Step 6 already accepted this frame — no further decisions needed
        obs = obs_by_index.get(frame.frame_index, {})
        history = frame.enhancement_history
        round_num = len(history) + 1

        if round_num > MAX_ENHANCEMENT_ROUNDS:
            decisions.append(AgentDecision(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                round_num=round_num,
                thought="Max enhancement rounds reached — stopping to avoid an infinite loop.",
                action="none",
                expects_improvement=False,
                valid=True,
            ))
            continue

        try:
            result = react_agent.decide(
                degradation_type=obs.get("degradation_type"),
                quality_score=obs.get("quality_score"),
                vlm_reasoning=obs.get("reasoning"),
                history=history,
                round_num=round_num,
            )
            action = result.get("action", "none")
            valid = action in TOOL_REGISTRY
            decisions.append(AgentDecision(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                round_num=round_num,
                thought=result.get("thought", ""),
                action=action if valid else "none",
                expects_improvement=bool(result.get("expects_improvement", False)),
                valid=valid,
            ))
        except (requests.RequestException, ValueError, TimeoutError) as exc:
            decisions.append(AgentDecision(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                round_num=round_num,
                thought=f"Agent call failed ({exc}); defaulting to no action.",
                action="none",
                expects_improvement=False,
                valid=False,
            ))

    return {
        **state,
        "enhancement_plan": [asdict(d) for d in decisions],
        "error": None,
    }


# --------------------------------------------------------------------------
# 4. Standalone smoke test — chains Step 1 -> 2 -> 3 -> 4
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step3_quality_assessment import quality_assessment_node, LocalVLMClient

    if len(sys.argv) < 2:
        print("Usage: python step4_react_agent.py <path_to_video>")
        sys.exit(1)

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    print(f"Sampled {state['frames_sampled']} frames — running VLM assessment...")
    state = quality_assessment_node(state, vlm_client=LocalVLMClient())

    print("Running ReAct agent decisions...")
    state = react_agent_node(state, agent=ReActAgent())

    for d in state["enhancement_plan"]:
        marker = "" if d["valid"] else "  [INVALID -> fell back to none]"
        print(
            f"  t={d['timestamp_sec']:6.2f}s  round={d['round_num']}  "
            f"action={d['action']:20s}  expects_improvement={d['expects_improvement']}{marker}"
        )
        print(f"      thought: {d['thought']}")