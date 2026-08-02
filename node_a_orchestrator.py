# node_a_orchestrator.py
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Dict, List

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import (
    ENHANCED_FRAME_FOLDER,
    EXTRACTED_FRAME_FOLDER,
    HTTP_TIMEOUT_SECONDS,
    NODE_A_PUBLIC_URL,
    NODE_B_URL,
    NODE_C_URL,
    UPLOAD_FOLDER,
)
from key_frame_extractor import KeyframeExtractor
from schemas import (
    EnhancementResult,
    ExtractedFrame,
    FrameReportEntry,
    JobProgress,
    JobStatus,
    VideoReport,
    VLMResult,
)

app = FastAPI(title="Node A - Orchestrator")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FRAME_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FRAME_FOLDER, exist_ok=True)

app.mount("/extracted_frames", StaticFiles(directory=EXTRACTED_FRAME_FOLDER), name="extracted_frames")
app.mount("/enhanced_frames", StaticFiles(directory=ENHANCED_FRAME_FOLDER), name="enhanced_frames")

_jobs: Dict[str, JobProgress] = {}
_extractor = KeyframeExtractor(output_folder=EXTRACTED_FRAME_FOLDER)


async def _call_node_b(client: httpx.AsyncClient, frame: ExtractedFrame) -> VLMResult:
    try:
        with open(frame.filepath, "rb") as f:
            files = {"image": (os.path.basename(frame.filepath), f.read(), "image/jpeg")}
        response = await client.post(
            f"{NODE_B_URL}/analyze",
            files=files,
            data={"frame_number": str(frame.frame_number)},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return VLMResult(**response.json())
    except Exception as exc:
        return VLMResult(
            frame_number=frame.frame_number,
            text_description="",
            overall_quality="Unknown",
            summary=f"Node B unavailable: {exc}",
        )


async def _call_node_c(client: httpx.AsyncClient, frame: ExtractedFrame) -> EnhancementResult:
    filename = os.path.splitext(os.path.basename(frame.filepath))[0]
    try:
        with open(frame.filepath, "rb") as f:
            files = {"image": (os.path.basename(frame.filepath), f.read(), "image/jpeg")}
        response = await client.post(
            f"{NODE_C_URL}/enhance",
            files=files,
            data={"filename": filename},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        enhanced_name = f"{filename}_enhanced.jpg"
        enhanced_path = os.path.join(ENHANCED_FRAME_FOLDER, enhanced_name)
        with open(enhanced_path, "wb") as out:
            out.write(response.content)

        return EnhancementResult(
            frame_number=frame.frame_number,
            enhanced_filepath=enhanced_path,
            enhanced_url=f"{NODE_A_PUBLIC_URL}/enhanced_frames/{enhanced_name}",
            upscale_factor=2.0,
        )
    except Exception as exc:
        return EnhancementResult(
            frame_number=frame.frame_number,
            enhanced_filepath="",
            enhanced_url="",
            upscale_factor=0.0,
            applied=False,
            skip_reason=str(exc),
        )


async def _process_frame(client: httpx.AsyncClient, frame: ExtractedFrame) -> FrameReportEntry:
    vlm_result, enhancement_result = await asyncio.gather(
        _call_node_b(client, frame),
        _call_node_c(client, frame),
    )
    return FrameReportEntry(frame=frame, vlm=vlm_result, enhancement=enhancement_result)


async def _run_pipeline(job_id: str, video_path: str):
    job = _jobs[job_id]
    try:
        job.status = JobStatus.EXTRACTING
        start_time = time.time()

        frames: List[ExtractedFrame] = []
        async for frame in _extractor.extract(video_path):
            frame.url = f"{NODE_A_PUBLIC_URL}/extracted_frames/{os.path.basename(frame.filepath)}"
            frames.append(frame)
            job.frames_extracted = len(frames)

        job.status = JobStatus.PROCESSING
        job.total_frames_estimate = len(frames)

        entries: List[FrameReportEntry] = []
        async with httpx.AsyncClient() as client:
            tasks = [_process_frame(client, frame) for frame in frames]
            for coro in asyncio.as_completed(tasks):
                entry = await coro
                entries.append(entry)
                job.frames_completed = len(entries)

        entries.sort(key=lambda e: e.frame.frame_number)

        job.report = VideoReport(
            video_name=os.path.basename(video_path),
            total_frames_in_video=len(frames),
            total_keyframes_extracted=len(entries),
            processing_time_seconds=round(time.time() - start_time, 2),
            entries=entries,
        )
        job.status = JobStatus.DONE
    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error_message = str(exc)


@app.post("/upload", response_model=JobProgress)
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")

    with open(video_path, "wb") as f:
        f.write(await file.read())

    _jobs[job_id] = JobProgress(job_id=job_id, status=JobStatus.PENDING)
    asyncio.create_task(_run_pipeline(job_id, video_path))
    return _jobs[job_id]


@app.get("/status/{job_id}", response_model=JobProgress)
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/report/{job_id}", response_model=VideoReport)
async def get_report(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.report is None:
        raise HTTPException(status_code=404, detail="Report not ready")
    return job.report

from agentic.supervisor import run_agentic_pipeline

@app.post("/agentic/upload", response_model=JobProgress)
async def upload_video_agentic(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")
    with open(video_path, "wb") as f:
        f.write(await file.read())
    _jobs[job_id] = JobProgress(job_id=job_id, status=JobStatus.PENDING)
    asyncio.create_task(run_agentic_pipeline(job_id, video_path))
    return _jobs[job_id]

@app.get("/report/{job_id}/pdf")
async def download_report_pdf(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.report is None or not job.report.report_pdf_path:
        raise HTTPException(status_code=404, detail="Report PDF not ready")
    return FileResponse(job.report.report_pdf_path, media_type="application/pdf",
                         filename=os.path.basename(job.report.report_pdf_path))