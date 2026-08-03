"""
agentic/state.py

Aligned to your real schemas.py — ExtractedFrame and VLMResult are the
actual Pydantic models, not a made-up dict shape.
"""

from typing import TypedDict, List, Dict, Optional, Literal, Annotated
import operator

from schemas import ExtractedFrame, VLMResult, AnomalyResult


class EnhancementRecord(TypedDict):
    tool: str
    quality_before: float
    quality_after: Optional[float]
    improved: Optional[bool]
    reasoning: str


class FrameState(TypedDict):
    job_id: str
    video_path: str

    original_frame: ExtractedFrame     # never mutated — source of truth for "before"
    keyframe_path: str                 # current working path (original, then enhanced)
    original_keyframe_path: str        # fixed copy of original_frame.filepath

    vlm_result: Optional[VLMResult]         # before-enhancement VLM report
    post_vlm_result: Optional[VLMResult]    # after-enhancement VLM report (this iteration)
    anomaly_result: Optional[AnomalyResult] # checked once, on the final (best-quality) frame

    selected_tool: Optional[
        Literal["real_esrgan", "opencv_denoise", "clahe", "sharpen", "none"]
    ]
    accepted: bool
    retry_count: int
    max_retries: int

    enhancement_history: Annotated[List[EnhancementRecord], operator.add]
    reasoning_trace: Annotated[List[str], operator.add]

    status: Literal[
        "assessing", "selecting", "enhancing", "validating", "accepted", "failed",
    ]
    error: Optional[str]