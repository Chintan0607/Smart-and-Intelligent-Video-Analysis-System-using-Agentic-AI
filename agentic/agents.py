"""
agentic/agents.py

Each function is an async LangGraph node: (FrameState) -> partial update.
Reuses your existing _call_node_b / _call_node_c from
node_a_orchestrator.py — no duplicate HTTP client code.
"""

import json
import logging
from typing import Dict

import httpx
from langchain_anthropic import ChatAnthropic

from .state import FrameState
from .quality import quality_score, improved, describe
from .local_enhancers import LOCAL_TOOLS
from schemas import EnhancementResult

logger = logging.getLogger("agentic.agents")

REACT_MODEL = "claude-sonnet-4-6"
_llm = ChatAnthropic(model=REACT_MODEL, temperature=0)


def _current_frame(state: FrameState):
    """ExtractedFrame pointing at whatever image is currently 'live' for this frame
    (original on first pass, enhanced output on retries) — needed because
    _call_node_b/_call_node_c both read frame.filepath."""
    return state["original_frame"].model_copy(update={"filepath": state["keyframe_path"]})


# ---------------------------------------------------------------------
# 2. Quality Assessment Agent — calls Node B (Qwen VLM)
# ---------------------------------------------------------------------
async def quality_assessment_agent(state: FrameState) -> Dict:
    from node_a_orchestrator import _call_node_b  # imported late to avoid circular import at module load

    async with httpx.AsyncClient() as client:
        report = await _call_node_b(client, _current_frame(state))

    logger.info(f"[quality] frame {report.frame_number}: {describe(report)}")
    return {
        "vlm_result": report,
        "status": "assessing",
        "reasoning_trace": [f"VLM report (frame {report.frame_number}): {describe(report)}"],
    }


# ---------------------------------------------------------------------
# 3. Enhancement Selection Agent — ReAct over the actual defects dict
# ---------------------------------------------------------------------
_SELECTION_SYSTEM_PROMPT = """You are the Enhancement Selection Agent in a video
quality pipeline. You receive a VLM quality report (overall_quality, a
per-category defects dict where each entry has present/confidence/severity/
evidence, and regeneration_recommended). Reason step by step, then respond
with ONLY a JSON object:
{"tool": "<real_esrgan|opencv_denoise|clahe|sharpen|none>", "reasoning": "<why>"}

Guidance (use judgement, do not follow blindly):
- blur / resolution / compression / artifacts defects, especially severe -> real_esrgan (GPU super-resolution/restoration)
- noise defect -> opencv_denoise
- lighting defect (poor exposure/contrast) -> clahe
- mild blur only, otherwise clean -> sharpen
- regeneration_recommended is False and no defects present -> none
Never pick a tool already tried and rejected for this frame (see history)."""


def _serialize_vlm(vlm) -> dict:
    return {
        "overall_quality": vlm.overall_quality,
        "regeneration_recommended": vlm.regeneration_recommended,
        "defects": {k: v.model_dump() for k, v in vlm.defects.items()},
    }


async def enhancement_selection_agent(state: FrameState) -> Dict:
    tried = [h["tool"] for h in state.get("enhancement_history", [])]
    user_msg = (
        f"VLM report: {json.dumps(_serialize_vlm(state['vlm_result']))}\n"
        f"Already tried and rejected: {tried}\n"
        f"Retry count: {state['retry_count']} / {state['max_retries']}"
    )
    response = await _llm.ainvoke([
        ("system", _SELECTION_SYSTEM_PROMPT),
        ("user", user_msg),
    ])
    raw = response.content
    try:
        parsed = json.loads(raw)
        tool_name = parsed.get("tool", "none")
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, TypeError):
        tool_name, reasoning = "none", f"Fallback: could not parse LLM output: {raw!r}"

    valid_tools = {"real_esrgan", "opencv_denoise", "clahe", "sharpen", "none"}
    if tool_name not in valid_tools or tool_name in tried:
        tool_name = "none"

    logger.info(f"[selection] chose '{tool_name}': {reasoning}")
    return {
        "selected_tool": tool_name,
        "status": "selecting",
        "reasoning_trace": [f"Selected '{tool_name}': {reasoning}"],
    }


# ---------------------------------------------------------------------
# 4. Enhancement Execution Agent
# ---------------------------------------------------------------------
async def enhancement_execution_agent(state: FrameState) -> Dict:
    tool_name = state["selected_tool"]
    if tool_name == "none":
        return {"status": "validating", "reasoning_trace": ["No enhancement applied."]}

    if tool_name == "real_esrgan":
        from node_a_orchestrator import _call_node_c  # Machine 4 (GPU) over HTTP

        async with httpx.AsyncClient() as client:
            result: EnhancementResult = await _call_node_c(client, _current_frame(state))

        if not result.applied:
            logger.warning(f"[execution] real_esrgan skipped: {result.skip_reason}")
            return {
                "status": "validating",
                "reasoning_trace": [f"real_esrgan unavailable: {result.skip_reason}"],
            }
        new_path = result.enhanced_filepath
    else:
        new_path = LOCAL_TOOLS[tool_name](state["keyframe_path"])

    logger.info(f"[execution] {tool_name} -> {new_path}")
    return {
        "keyframe_path": new_path,
        "status": "enhancing",
        "reasoning_trace": [f"Executed {tool_name} -> {new_path}"],
    }


# ---------------------------------------------------------------------
# 5. Validation Agent — calls Node B again, compares before vs after
# ---------------------------------------------------------------------
async def validation_agent(state: FrameState) -> Dict:
    if state["selected_tool"] == "none":
        return {
            "accepted": True,
            "status": "accepted",
            "enhancement_history": [{
                "tool": "none",
                "quality_before": quality_score(state["vlm_result"]),
                "quality_after": quality_score(state["vlm_result"]),
                "improved": True,
                "reasoning": "Frame already acceptable; no enhancement attempted.",
            }],
        }

    from node_a_orchestrator import _call_node_b

    async with httpx.AsyncClient() as client:
        new_report = await _call_node_b(client, _current_frame(state))

    did_improve = improved(state["vlm_result"], new_report)
    record: Dict = {
        "tool": state["selected_tool"],
        "quality_before": quality_score(state["vlm_result"]),
        "quality_after": quality_score(new_report),
        "improved": did_improve,
        "reasoning": describe(new_report),
    }
    logger.info(f"[validation] {record}")

    next_retry = state["retry_count"] + 1
    exhausted = next_retry >= state.get("max_retries", 3)

    return {
        "post_vlm_result": new_report,
        "accepted": did_improve,
        "retry_count": next_retry,
        "status": "accepted" if (did_improve or exhausted) else "validating",
        "enhancement_history": [record],
        "reasoning_trace": [
            f"Validation: improved={did_improve}, retry={next_retry}/{state['max_retries']}"
        ],
    }
