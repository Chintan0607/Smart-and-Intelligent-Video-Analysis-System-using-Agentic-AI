"""
agentic/graph.py

Per-frame async LangGraph: Quality -> Select -> Execute -> Validate -> (retry|accept)
"""

from langgraph.graph import StateGraph, END

from .state import FrameState
from .agents import (
    quality_assessment_agent,
    enhancement_selection_agent,
    enhancement_execution_agent,
    validation_agent,
)


def _route_after_validation(state: FrameState) -> str:
    if state["accepted"]:
        return "end"
    if state["retry_count"] >= state.get("max_retries", 3):
        return "end"   # retries exhausted — force stop, keep best-effort result
    return "retry"


def build_frame_graph():
    graph = StateGraph(FrameState)

    graph.add_node("quality_assessment", quality_assessment_agent)
    graph.add_node("enhancement_selection", enhancement_selection_agent)
    graph.add_node("enhancement_execution", enhancement_execution_agent)
    graph.add_node("validation", validation_agent)

    graph.set_entry_point("quality_assessment")
    graph.add_edge("quality_assessment", "enhancement_selection")
    graph.add_edge("enhancement_selection", "enhancement_execution")
    graph.add_edge("enhancement_execution", "validation")

    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"retry": "enhancement_selection", "end": END},
    )

    return graph.compile()


FRAME_GRAPH = build_frame_graph()
