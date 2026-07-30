"""LangGraph workflow to select GAN candidate frames from frame quality evaluations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END


@dataclass
class GANSelectionState:
    raw_frame_evaluations: List[Dict[str, Any]]
    frames_by_number: Dict[int, Any]
    frame_scores: Dict[int, float] = field(default_factory=dict)
    selected_frame_numbers: List[int] = field(default_factory=list)
    selected_frames: List[Any] = field(default_factory=list)
    recommended_gan: Optional[str] = None
    decision_trace: List[str] = field(default_factory=list)
    max_candidates: int = 5
    min_score: float = 0.25


def _compute_frame_score(evaluation: Dict[str, Any]) -> float:
    defects = evaluation.get("defects", {})
    weights = {
        "Blur": 1.0,
        "Noise": 1.2,
        "Compression": 1.0,
        "Lighting": 0.8,
    }

    score = 0.0
    for defect_name, weight in weights.items():
        present = defects.get(defect_name, {}).get("present", False)
        if present:
            score += weight

    overall_quality = evaluation.get("overall_quality", "").lower()
    if overall_quality == "needs review":
        score += 0.3
    if overall_quality == "good":
        score -= 0.2

    summary = evaluation.get("summary", "").lower()
    if "lighting" in summary and "noise" in summary:
        score += 0.1

    return max(0.0, min(score / 4.0, 1.0))


def _select_gan_tool(evaluation: Dict[str, Any]) -> str:
    defects = evaluation.get("defects", {})
    if defects.get("Blur", {}).get("present"):
        return "deblurgan_v2"
    if defects.get("Noise", {}).get("present") or defects.get("Compression", {}).get("present"):
        return "restormer"
    if defects.get("Lighting", {}).get("present"):
        return "histogram_equalization"
    return "real_esrgan"


def evaluate_frames_node(state: GANSelectionState) -> GANSelectionState:
    state.frame_scores = {}

    for evaluation in state.raw_frame_evaluations:
        frame_number = int(evaluation.get("frame_number", -1))
        if frame_number < 0:
            continue

        score = _compute_frame_score(evaluation)
        state.frame_scores[frame_number] = score
        state.decision_trace.append(
            f"Frame {frame_number} score={score:.2f}"
        )

    return state


def choose_candidates_node(state: GANSelectionState) -> GANSelectionState:
    sorted_frames = sorted(
        state.frame_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    candidates = [frame_number for frame_number, score in sorted_frames if score >= state.min_score]
    if not candidates and sorted_frames:
        candidates = [sorted_frames[0][0]]
        state.decision_trace.append(
            f"No frame met min_score={state.min_score}; defaulting to highest-scored frame {candidates[0]}"
        )

    state.selected_frame_numbers = candidates[: state.max_candidates]
    state.decision_trace.append(
        f"Selected frame numbers for GAN enhancement: {state.selected_frame_numbers}"
    )
    return state


def select_gan_tool_node(state: GANSelectionState) -> GANSelectionState:
    if not state.selected_frame_numbers:
        return state

    top_frame_number = state.selected_frame_numbers[0]
    top_evaluation = next(
        (item for item in state.raw_frame_evaluations if int(item.get("frame_number", -1)) == top_frame_number),
        None,
    )

    if top_evaluation is not None:
        state.recommended_gan = _select_gan_tool(top_evaluation)
        state.decision_trace.append(
            f"Recommended GAN tool based on frame {top_frame_number}: {state.recommended_gan}"
        )
    else:
        state.recommended_gan = "real_esrgan"
        state.decision_trace.append(
            "Could not find top frame evaluation; defaulting GAN tool to real_esrgan"
        )

    return state


def finalize_selection_node(state: GANSelectionState) -> GANSelectionState:
    state.selected_frames = []

    for frame_number in state.selected_frame_numbers:
        frame = state.frames_by_number.get(frame_number)
        if frame is not None:
            state.selected_frames.append(frame)
        else:
            state.decision_trace.append(
                f"Frame {frame_number} not found in frames_by_number; skipping image payload"
            )

    return state


def create_gan_selection_workflow() -> StateGraph:
    workflow = StateGraph(GANSelectionState)
    workflow.add_node("evaluate_frames", evaluate_frames_node)
    workflow.add_node("choose_candidates", choose_candidates_node)
    workflow.add_node("select_gan_tool", select_gan_tool_node)
    workflow.add_node("finalize_selection", finalize_selection_node)

    workflow.add_edge(START, "evaluate_frames")
    workflow.add_edge("evaluate_frames", "choose_candidates")
    workflow.add_edge("choose_candidates", "select_gan_tool")
    workflow.add_edge("select_gan_tool", "finalize_selection")
    workflow.add_edge("finalize_selection", END)

    return workflow.compile()


def run_gan_selection_workflow(
    raw_frame_evaluations: List[Dict[str, Any]],
    frames_by_number: Optional[Dict[int, Any]] = None,
    max_candidates: int = 5,
    min_score: float = 0.25,
) -> GANSelectionState:
    state = GANSelectionState(
        raw_frame_evaluations=raw_frame_evaluations,
        frames_by_number=frames_by_number or {},
        max_candidates=max_candidates,
        min_score=min_score,
    )

    workflow = create_gan_selection_workflow()
    result = workflow.invoke(state)
    return result


# Example usage:
# selected_state = run_gan_selection_workflow(
#     raw_frame_evaluations=json_payload["raw_frame_evaluations"],
#     frames_by_number=extracted_frames_by_number,
#     max_candidates=6,
# )
# print(selected_state.selected_frame_numbers)
# print(selected_state.recommended_gan)
