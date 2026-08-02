"""
agentic/supervisor.py

Whole-video orchestration for the agentic pipeline. This does NOT
replace node_a_orchestrator.py's existing /upload + _run_pipeline —
it's a second pipeline (`run_agentic_pipeline`) wired to a new
/agentic/upload route, reusing the same `_jobs` dict, `_extractor`,
and JobProgress/VideoReport schemas so /status/{job_id} and
/report/{job_id} keep working unchanged for either pipeline.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List

from .graph import FRAME_GRAPH
from .report_generator import generate_pdf_report
from schemas import (
    ExtractedFrame, FrameReportEntry, EnhancementResult, JobStatus, VideoReport,
)

logger = logging.getLogger("agentic.supervisor")

MAX_ENHANCEMENT_RETRIES = int(os.getenv("MAX_ENHANCEMENT_RETRIES", "3"))
REPORT_FOLDER = os.getenv("REPORT_FOLDER", "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)


def _initial_frame_state(job_id: str, video_path: str, frame: ExtractedFrame) -> Dict:
    return {
        "job_id": job_id,
        "video_path": video_path,
        "original_frame": frame,
        "keyframe_path": frame.filepath,
        "original_keyframe_path": frame.filepath,
        "vlm_result": None,
        "post_vlm_result": None,
        "selected_tool": None,
        "accepted": False,
        "retry_count": 0,
        "max_retries": MAX_ENHANCEMENT_RETRIES,
        "enhancement_history": [],
        "reasoning_trace": [],
        "status": "assessing",
        "error": None,
    }


async def _process_frame_agentic(job_id: str, video_path: str, frame: ExtractedFrame) -> Dict:
    initial_state = _initial_frame_state(job_id, video_path, frame)
    try:
        final_state = await FRAME_GRAPH.ainvoke(initial_state)
    except Exception as e:
        logger.exception(f"[supervisor] frame {frame.frame_number} failed: {e}")
        final_state = {**initial_state, "status": "failed", "error": str(e), "accepted": False}
    return final_state


def _to_report_entry(frame: ExtractedFrame, final_state: Dict) -> FrameReportEntry:
    """Builds a FrameReportEntry. `enhancement` uses your existing EnhancementResult
    shape for the LAST tool applied; the fuller retry/tool history goes in the
    extra fields you'll add to FrameReportEntry (see schemas.py addition note)."""
    last_tool = final_state["enhancement_history"][-1]["tool"] if final_state["enhancement_history"] else "none"
    enhancement = EnhancementResult(
        frame_number=frame.frame_number,
        enhanced_filepath=final_state["keyframe_path"],
        enhanced_url="",  # fill in if/when you serve enhanced files over NODE_A_PUBLIC_URL
        upscale_factor=4.0 if last_tool == "real_esrgan" else 1.0,
        applied=last_tool != "none",
        skip_reason=final_state.get("error"),
    )
    entry = FrameReportEntry(
        frame=frame,
        vlm=final_state.get("post_vlm_result") or final_state["vlm_result"],
        enhancement=enhancement,
    )
    # extra fields — requires the schemas.py addition below
    entry.retries = final_state["retry_count"]
    entry.history = final_state.get("enhancement_history", [])
    entry.vlm_before = final_state["vlm_result"]
    return entry


async def run_agentic_pipeline(job_id: str, video_path: str):
    """Entry point called from the new /agentic/upload route in node_a_orchestrator.py."""
    from node_a_orchestrator import _jobs, _extractor  # late import: reuse existing job store + extractor

    job = _jobs[job_id]
    start_time = time.time()
    try:
        job.status = JobStatus.EXTRACTING
        frames: List[ExtractedFrame] = []
        async for frame in _extractor.extract(video_path):
            frames.append(frame)
            job.frames_extracted = len(frames)

        job.status = JobStatus.PROCESSING
        job.total_frames_estimate = len(frames)

        entries: List[FrameReportEntry] = []
        tasks = [_process_frame_agentic(job_id, video_path, f) for f in frames]
        results_by_frame = {}
        for coro in asyncio.as_completed(tasks):
            final_state = await coro
            results_by_frame[final_state["original_frame"].frame_number] = final_state
            job.frames_completed = len(results_by_frame)

        for frame in frames:
            final_state = results_by_frame[frame.frame_number]
            entries.append(_to_report_entry(frame, final_state))

        entries.sort(key=lambda e: e.frame.frame_number)
        total_seconds = round(time.time() - start_time, 2)

        job.report = VideoReport(
            video_name=os.path.basename(video_path),
            total_frames_in_video=len(frames),
            total_keyframes_extracted=len(entries),
            processing_time_seconds=total_seconds,
            entries=entries,
        )

        report_pdf_path = os.path.join(REPORT_FOLDER, f"{job_id}_report.pdf")
        generate_pdf_report(job.report, report_pdf_path)
        job.report.report_pdf_path = report_pdf_path  # requires schemas.py addition

        job.status = JobStatus.DONE
        logger.info(f"[supervisor] job {job_id} done in {total_seconds}s, report at {report_pdf_path}")

    except Exception as exc:
        logger.exception(f"[supervisor] job {job_id} failed: {exc}")
        job.status = JobStatus.ERROR
        job.error_message = str(exc)
