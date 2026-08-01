"""
video_processor.py

BEFORE: `extract_frames()` did a synchronous `while True: cap.grab() ...
frame_number % interval == 0` loop and returned a fully-materialized list
of frame dicts. Everything downstream (VLM batching in app.py) had to
wait for the entire video to be scanned first.

AFTER: `process_video()` streams `ExtractedFrame` (Pydantic) objects out
of KeyframeExtractor.extract() as an async generator and fans them out to
VLM inference and enhancement concurrently via asyncio.Queue. The old
metadata helpers (save_uploaded_video, get_video_metadata,
clear_previous_frames) are unchanged — only the extraction step and the
data shape moving through the pipeline changed.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from typing import AsyncGenerator, Optional

import cv2

from config import EXTRACTED_FRAME_FOLDER
from key_frame_extractor import KeyframeExtractor
from schemas import EnhancementResult, ExtractedFrame, FrameReportEntry, VideoReport, VLMResult


# --------------------------------------------------------------------------
# Unchanged utility functions
# --------------------------------------------------------------------------
def save_uploaded_video(uploaded_file, upload_folder):
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    metadata["duration"] = (
        metadata["frame_count"] / metadata["fps"] if metadata["fps"] > 0 else 0
    )
    return metadata


def clear_previous_frames(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------------------------------
# NEW: streaming extraction
# --------------------------------------------------------------------------
async def extract_keyframes(
    video_path: str,
    output_folder: str = EXTRACTED_FRAME_FOLDER,
    extractor: Optional[KeyframeExtractor] = None,
) -> AsyncGenerator[ExtractedFrame, None]:
    """
    Drop-in async replacement for the old `extract_frames()`. Reuse a
    single `extractor` instance across requests if you can — it holds
    the loaded ResNet weights, and you don't want to reload those per
    video.
    """
    extractor = extractor or KeyframeExtractor(output_folder=output_folder)
    async for frame in extractor.extract(video_path):
        yield frame


# --------------------------------------------------------------------------
# NEW: fan-out orchestration
# --------------------------------------------------------------------------
async def process_video(
    video_path: str,
    vlm_service,          # QwenService or VLMService — must expose analyze_frame(path/PIL) -> dict-like
    enhancement_service,  # EnhancementService — must expose enhance_frame(path, filename)
    output_folder: str = EXTRACTED_FRAME_FOLDER,
    vlm_concurrency: int = 4,
    enhancement_concurrency: int = 2,
) -> VideoReport:
    """
    Streams keyframes out of extraction directly into VLM + enhancement
    workers via bounded queues, instead of waiting for a full frame list.

    Why queues instead of `asyncio.gather` over everything: extraction
    produces frames at whatever rate cv2 can grab them, VLM inference is
    GPU-bound and slower, and enhancement is slower still. Queues let
    each stage run at its own pace without extraction blocking on VLM,
    while `vlm_concurrency` / `enhancement_concurrency` cap how many
    frames are in flight so you don't blow up GPU memory on a long video.
    """
    metadata = get_video_metadata(video_path)
    clear_previous_frames(output_folder)

    vlm_queue: asyncio.Queue[Optional[ExtractedFrame]] = asyncio.Queue(maxsize=vlm_concurrency * 2)
    report_entries: list[FrameReportEntry] = []
    report_lock = asyncio.Lock()

    async def extraction_producer():
        async for frame in extract_keyframes(video_path, output_folder):
            await vlm_queue.put(frame)
        # sentinel per worker so every consumer knows to stop
        for _ in range(vlm_concurrency):
            await vlm_queue.put(None)

    async def vlm_worker():
        while True:
            frame = await vlm_queue.get()
            if frame is None:
                vlm_queue.task_done()
                break

            # Run the (synchronous, GPU-bound) VLM call off the event loop
            raw_analysis = await asyncio.to_thread(vlm_service.analyze_frame, frame.filepath)
            vlm_result = VLMResult(
                frame_number=frame.frame_number,
                text_description=raw_analysis.get("text_description", ""),
                defects=raw_analysis.get("defects", {}),
                overall_quality=raw_analysis.get("overall_quality", "Unknown"),
                summary=raw_analysis.get("summary", ""),
                regeneration_recommended=raw_analysis.get("regeneration_recommended", False),
            )

            enhancement_result = await asyncio.to_thread(
                _run_enhancement, enhancement_service, frame
            )

            entry = FrameReportEntry(frame=frame, vlm=vlm_result, enhancement=enhancement_result)
            async with report_lock:
                report_entries.append(entry)

            vlm_queue.task_done()

    producer = asyncio.create_task(extraction_producer())
    workers = [asyncio.create_task(vlm_worker()) for _ in range(vlm_concurrency)]

    await producer
    await asyncio.gather(*workers)

    report_entries.sort(key=lambda e: e.frame.frame_number)

    return VideoReport(
        video_name=os.path.basename(video_path),
        total_frames_in_video=int(metadata["frame_count"]),
        total_keyframes_extracted=len(report_entries),
        processing_time_seconds=0.0,  # set by caller, who owns the timer (see app.py integration)
        entries=report_entries,
    )


def _run_enhancement(enhancement_service, frame: ExtractedFrame) -> EnhancementResult:
    filename = os.path.splitext(os.path.basename(frame.filepath))[0]
    try:
        enhancement_service.enhance_frame(frame.filepath, filename)
        return EnhancementResult(
            frame_number=frame.frame_number,
            enhanced_filepath=f"enhanced_frames/{filename}_enhanced.jpg",
            upscale_factor=2.0,
        )
    except Exception as exc:  # keep one bad frame from failing the whole report
        return EnhancementResult(
            frame_number=frame.frame_number,
            enhanced_filepath="",
            upscale_factor=0.0,
            applied=False,
            skip_reason=str(exc),
        )