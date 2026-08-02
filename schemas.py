# schemas.py
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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
    retries: int = 0
    history: list[dict] = Field(default_factory=list)
    vlm_before: Optional[VLMResult] = None


class VideoReport(BaseModel):
    video_name: str
    total_frames_in_video: int
    total_keyframes_extracted: int
    processing_time_seconds: float
    entries: list[FrameReportEntry]
    report_pdf_path: Optional[str] = None

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