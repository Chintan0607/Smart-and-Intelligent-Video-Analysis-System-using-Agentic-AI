"""
agent_orchestrator.py

The Agentic Orchestrator. Extends Node A: reuses its KeyframeExtractor
and its httpx-based calls to Node B/Node C, but replaces the old
single-shot "enhance once and record the result" flow with a bounded
validation loop per frame.

State machine per frame (max_iterations, default 3):

    ┌─────────────────────────────────────────────┐
    │ 1. Send current frame to Node B (VLM)        │
    │ 2. regeneration_recommended == False? → BREAK│
    │ 3. Router reads `defects` → picks a tool      │
    │ 4. Apply tool (Node C or local OpenCV)        │
    │    → new temp frame, loop back to 1           │
    └─────────────────────────────────────────────┘

Add these to config.py:
    AGENTIC_FRAME_FOLDER = "agentic_frames"
    MAX_AGENT_ITERATIONS = 3
    AGENT_FRAME_CONCURRENCY = 3
    PDF_REPORT_FOLDER = "reports"
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Dict, List, Optional

import cv2
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    AGENT_FRAME_CONCURRENCY,
    AGENTIC_FRAME_FOLDER,
    EXTRACTED_FRAME_FOLDER,
    HTTP_TIMEOUT_SECONDS,
    MAX_AGENT_ITERATIONS,
    NODE_A_PUBLIC_URL,
    NODE_B_URL,
    NODE_C_URL,
    PDF_REPORT_FOLDER,
    UPLOAD_FOLDER,
)
from key_frame_extractor import KeyframeExtractor
from opencv_tools import apply_local_tool
from pdf_report_generator import generate_pdf_report  # NOTE: fixed from `pdf_report_generator` (file doesn't exist under that name — was breaking startup)
from schemas import (
    AgentFrameResult,
    AgentRunReport,
    DefectDetail,
    ExtractedFrame,
    IterationRecord,
    JobProgress,
    JobStatus,
    ToolType,
    VLMResult,
)

app = FastAPI(title="Agentic Orchestrator")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FRAME_FOLDER, exist_ok=True)
os.makedirs(AGENTIC_FRAME_FOLDER, exist_ok=True)
os.makedirs(PDF_REPORT_FOLDER, exist_ok=True)

app.mount("/extracted_frames", StaticFiles(directory=EXTRACTED_FRAME_FOLDER), name="extracted_frames")
app.mount("/agentic_frames", StaticFiles(directory=AGENTIC_FRAME_FOLDER), name="agentic_frames")

_jobs: Dict[str, JobProgress] = {}
_extractor = KeyframeExtractor(output_folder=EXTRACTED_FRAME_FOLDER)


# --------------------------------------------------------------------------
# Step 1: VLM call (Node B)
# --------------------------------------------------------------------------
async def _call_node_b(client: httpx.AsyncClient, frame_path: str, frame_number: int) -> VLMResult:
    with open(frame_path, "rb") as f:
        files = {"image": (os.path.basename(frame_path), f.read(), "image/jpeg")}
    response = await client.post(
        f"{NODE_B_URL}/analyze",
        files=files,
        data={"frame_number": str(frame_number)},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return VLMResult(**response.json())


# --------------------------------------------------------------------------
# Step 3: Router — defects -> tool
# --------------------------------------------------------------------------
def decide_tool(defects: Dict[str, DefectDetail], attempted_tools: List[ToolType]) -> ToolType:
    """
    Deterministic priority router over the VLM's defect keys/evidence.
    """
    active_defects = {name.lower(): detail for name, detail in defects.items() if detail.present}

    def _matches(keywords: list[str]) -> bool:
        return any(keyword in name or keyword in active_defects[name].evidence.lower()
                   for name in active_defects for keyword in keywords)

    # 1. LIGHTING FIRST: Fix exposure issues before sending to the GAN. 
    # The GAN cannot upscale details it cannot see in the dark.
    if _matches(["lighting", "low_light", "dark", "underexpos", "contrast", "dim"]):
        if ToolType.OPENCV_CLAHE not in attempted_tools:
            return ToolType.OPENCV_CLAHE

    # 2. DEFAULT TO GAN: For blur, motion, noise, compression, and pixelation, 
    # run Real-ESRGAN to safely denoise and reconstruct the image.
    if _matches(["blur", "motion", "shake", "pixelat", "resolution", "low_res", "compression", "noise", "grain"]):
        if ToolType.GAN_ESRGAN not in attempted_tools:
            return ToolType.GAN_ESRGAN

    # Fallback: If GAN was already tried and the VLM still flags an issue, 
    # return the GAN again. The agent loop will automatically hit max_iterations (3) and safely stop.
    return ToolType.GAN_ESRGAN


# --------------------------------------------------------------------------
# Step 4: Apply tool
# --------------------------------------------------------------------------
async def _apply_gan_tool(
    client: httpx.AsyncClient, frame_path: str, frame_number: int, iteration: int
) -> str:
    filename = f"frame_{frame_number:06d}_iter{iteration}_gan"
    with open(frame_path, "rb") as f:
        files = {"image": (os.path.basename(frame_path), f.read(), "image/jpeg")}
    response = await client.post(
        f"{NODE_C_URL}/enhance",
        files=files,
        data={"filename": filename},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    output_path = os.path.join(AGENTIC_FRAME_FOLDER, f"{filename}_enhanced.jpg")
    with open(output_path, "wb") as out:
        out.write(response.content)
    return output_path


def _apply_opencv_tool(frame_path: str, frame_number: int, iteration: int, tool: ToolType) -> str:
    image = cv2.imread(frame_path)
    result = apply_local_tool(image, tool)

    output_path = os.path.join(
        AGENTIC_FRAME_FOLDER, f"frame_{frame_number:06d}_iter{iteration}_{tool.value}.jpg"
    )
    cv2.imwrite(output_path, result)
    return output_path


# --------------------------------------------------------------------------
# The validation loop, per frame
# --------------------------------------------------------------------------
async def run_agent_on_frame(
    client: httpx.AsyncClient,
    frame: ExtractedFrame,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> AgentFrameResult:
    current_path = frame.filepath
    iterations: List[IterationRecord] = []
    attempted_tools: List[ToolType] = []
    converged = False

    for iteration_number in range(1, max_iterations + 1):
        vlm_result = await _call_node_b(client, current_path, frame.frame_number)
        is_last_allowed_pass = iteration_number == max_iterations
        should_stop = (not vlm_result.regeneration_recommended) or is_last_allowed_pass

        if should_stop:
            iterations.append(
                IterationRecord(
                    iteration_number=iteration_number,
                    frame_path_evaluated=current_path,
                    vlm_summary=vlm_result.summary,
                    defects_detected=[name for name, d in vlm_result.defects.items() if d.present],
                    regeneration_recommended=vlm_result.regeneration_recommended,
                    tool_used=ToolType.NONE,
                    frame_path_after_tool=None,
                )
            )
            converged = not vlm_result.regeneration_recommended
            break

        tool = decide_tool(vlm_result.defects, attempted_tools)
        attempted_tools.append(tool)

        if tool == ToolType.GAN_ESRGAN:
            new_path = await _apply_gan_tool(client, current_path, frame.frame_number, iteration_number)
        else:
            new_path = await asyncio.to_thread(
                _apply_opencv_tool, current_path, frame.frame_number, iteration_number, tool
            )

        iterations.append(
            IterationRecord(
                iteration_number=iteration_number,
                frame_path_evaluated=current_path,
                vlm_summary=vlm_result.summary,
                defects_detected=[name for name, d in vlm_result.defects.items() if d.present],
                regeneration_recommended=vlm_result.regeneration_recommended,
                tool_used=tool,
                frame_path_after_tool=new_path,
            )
        )
        current_path = new_path

    return AgentFrameResult(
        frame_number=frame.frame_number,
        original_filepath=frame.filepath,
        final_filepath=current_path,
        iterations=iterations,
        converged=converged,
        total_tool_applications=sum(1 for rec in iterations if rec.tool_used != ToolType.NONE),
    )


# --------------------------------------------------------------------------
# Full pipeline: extract -> agent loop per frame (bounded concurrency) -> PDF
# --------------------------------------------------------------------------
async def _run_agentic_pipeline(job_id: str, video_path: str):
    job = _jobs[job_id]
    try:
        job.status = JobStatus.EXTRACTING
        start_time = time.time()

        frames: List[ExtractedFrame] = []
        async for frame in _extractor.extract(video_path):
            frames.append(frame)
            job.frames_extracted = len(frames)

        job.status = JobStatus.PROCESSING
        job.total_frames_estimate = len(frames)

        semaphore = asyncio.Semaphore(AGENT_FRAME_CONCURRENCY)
        frame_results: List[Optional[AgentFrameResult]] = [None] * len(frames)

        async def _bounded_run(index: int, frame: ExtractedFrame, client: httpx.AsyncClient):
            async with semaphore:
                frame_results[index] = await run_agent_on_frame(client, frame)
                job.frames_completed += 1

        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *(_bounded_run(i, frame, client) for i, frame in enumerate(frames))
            )

        agent_report = AgentRunReport(
            video_name=os.path.basename(video_path),
            total_frames_processed=len(frames),
            total_processing_time_seconds=round(time.time() - start_time, 2),
            max_iterations=MAX_AGENT_ITERATIONS,
            frame_results=[r for r in frame_results if r is not None],
        )

        pdf_filename = f"{job_id}_report.pdf"
        pdf_path = os.path.join(PDF_REPORT_FOLDER, pdf_filename)
        generate_pdf_report(agent_report, output_path=pdf_path)

        job.pdf_path = pdf_path
        job.status = JobStatus.DONE
    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error_message = str(exc)


# --------------------------------------------------------------------------
# FastAPI endpoints
# --------------------------------------------------------------------------
@app.post("/agentic/upload", response_model=JobProgress)
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{file.filename}")

    with open(video_path, "wb") as f:
        f.write(await file.read())

    _jobs[job_id] = JobProgress(job_id=job_id, status=JobStatus.PENDING)
    asyncio.create_task(_run_agentic_pipeline(job_id, video_path))
    return _jobs[job_id]


@app.get("/agentic/status/{job_id}", response_model=JobProgress)
async def get_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/agentic/report/{job_id}/pdf")
async def get_pdf_report(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF report not ready")
    return FileResponse(job.pdf_path, media_type="application/pdf", filename=os.path.basename(job.pdf_path))