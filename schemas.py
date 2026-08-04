"""
schemas.py

Superset of the prior shared schema file. Adds the agentic
validation-loop models (ToolType, IterationRecord, AgentFrameResult,
AgentRunReport) on top of the existing extraction/VLM/enhancement/job
models — nothing pre-existing was removed, so this is a drop-in
replacement across all three nodes.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
class ExtractionMethod(str, Enum):
    SHOT_BOUNDARY = "shot_boundary"
    SEMANTIC_NOVELTY = "semantic_novelty"
    FORCED_INTERVAL = "forced_interval"


class FrameMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    frame_number: int
    timestamp_sec: float = Field(ge=0.0)
    shot_id: int = Field(ge=0)
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    similarity_to_previous: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ExtractedFrame(BaseModel):
    frame_number: int
    filepath: str
    url: Optional[str] = None
    metadata: FrameMetadata


# --------------------------------------------------------------------------
# VLM
# --------------------------------------------------------------------------
class DefectDetail(BaseModel):
    present: bool
    confidence: str
    severity: str
    evidence: str


class VLMResult(BaseModel):
    frame_number: int
    text_description: str
    defects: dict[str, DefectDetail] = Field(default_factory=dict)
    overall_quality: str
    summary: str
    regeneration_recommended: bool = False


# --------------------------------------------------------------------------
# Enhancement (single-shot, non-agentic path — kept for backward compat)
# --------------------------------------------------------------------------
class EnhancementResult(BaseModel):
    frame_number: int
    enhanced_filepath: str
    enhanced_url: str
    upscale_factor: float
    applied: bool = True
    skip_reason: Optional[str] = None


class FrameReportEntry(BaseModel):
    frame: ExtractedFrame
    vlm: VLMResult
    enhancement: Optional[EnhancementResult] = None


class VideoReport(BaseModel):
    video_name: str
    total_frames_in_video: int
    total_keyframes_extracted: int
    processing_time_seconds: float
    entries: list[FrameReportEntry]


# --------------------------------------------------------------------------
# Job tracking (shared by both the single-shot and agentic pipelines)
# --------------------------------------------------------------------------
class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    frames_extracted: int = 0
    frames_completed: int = 0
    total_frames_estimate: Optional[int] = None
    error_message: Optional[str] = None
    report: Optional[VideoReport] = None
    pdf_path: Optional[str] = None


# --------------------------------------------------------------------------
# NEW: Agentic validation loop
# --------------------------------------------------------------------------
class ToolType(str, Enum):
    """What the agent decided to apply for a given iteration."""

    NONE = "none"                    # this record is a pure VLM evaluation, no tool applied
    GAN_ESRGAN = "gan_realesrgan"     # Node C — resolution / pixelation
    OPENCV_UNSHARP = "opencv_unsharp_mask"   # motion blur
    OPENCV_LAPLACIAN = "opencv_laplacian"    # motion blur (alternate/stronger)
    OPENCV_CLAHE = "opencv_clahe"     # low light / low contrast


class IterationRecord(BaseModel):
    """
    One pass through the loop: Step 1 (VLM eval) plus, if it wasn't the
    terminal pass, the Step 3/4 tool decision and application.
    """

    model_config = ConfigDict(frozen=True)

    iteration_number: int = Field(ge=1)
    frame_path_evaluated: str
    vlm_summary: str
    defects_detected: list[str] = Field(default_factory=list)
    regeneration_recommended: bool
    tool_used: ToolType = ToolType.NONE
    frame_path_after_tool: Optional[str] = None


class AgentFrameResult(BaseModel):
    """Everything the agent did for one keyframe."""

    frame_number: int
    original_filepath: str
    final_filepath: str
    iterations: list[IterationRecord]
    converged: bool  # True if it exited because quality was accepted, False if it hit max_iterations
    total_tool_applications: int

    @property
    def before_summary(self) -> str:
        return self.iterations[0].vlm_summary if self.iterations else ""

    @property
    def after_summary(self) -> str:
        return self.iterations[-1].vlm_summary if self.iterations else ""

    @property
    def tools_used(self) -> list[ToolType]:
        return [rec.tool_used for rec in self.iterations if rec.tool_used != ToolType.NONE]


class AgentRunReport(BaseModel):
    video_name: str
    total_frames_processed: int
    total_processing_time_seconds: float
    max_iterations: int
    frame_results: list[AgentFrameResult]