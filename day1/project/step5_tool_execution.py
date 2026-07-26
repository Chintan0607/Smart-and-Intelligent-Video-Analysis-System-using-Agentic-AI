"""
Step 5: Tool Execution Layer
==============================

Purpose
-------
Executes whatever Step 4's ReAct agent decided for each frame. Two tiers:

1. TRADITIONAL TOOLS (sharpen, denoise_cv2, histogram_equalization) —
   fully implemented here with OpenCV. These are real, working
   enhancements today, no external weights needed.

2. DEEP-LEARNING TOOLS (real_esrgan, deblurgan_v2, restormer, swinir) —
   STUBBED for now, since the model weights aren't downloaded yet. Each
   stub:
       - does NOT silently pass the frame through unchanged pretending
         it did something (that would corrupt Step 6's validation loop
         with false signal)
       - clearly logs `stub_only=True` on the result and in the frame's
         enhancement_history (e.g. "real_esrgan[STUB]") so Step 6 and any
         report can tell real enhancement from a placeholder
       - applies a tiny, honest, real OpenCV substitute where a
         reasonable one exists (e.g. real_esrgan's stub does a basic
         cv2 upscale — NOT true super-resolution, just so the pipeline
         has *something* to validate rather than a no-op), so testing
         the graph's control flow doesn't require the GPU models yet.

Swapping in real models later
------------------------------
Each DL tool has a matching `run_<tool>()` function with a single,
consistent signature: `(image: np.ndarray) -> np.ndarray`. To wire in the
real model once weights are downloaded, replace only that function's body
— nothing else in this file, in Step 4, or in the graph wiring needs to
change. Just also flip that tool's entry in `STUB_TOOLS` to remove it
from the stub set.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional

from step1_video_input import PipelineState
from step2_frame_extraction import PipelineStateWithFrames, FrameRecord

# --------------------------------------------------------------------------
# 1. Tools currently backed by a stub (no real weights yet)
# --------------------------------------------------------------------------
# Remove an entry here once you've wired in the real model for that tool.

STUB_TOOLS = {"deblurgan_v2", "restormer", "swinir"}  # real_esrgan removed — now dynamically checked per call


@dataclass
class ExecutionResult:
    frame_index: int
    tool_applied: str
    stub_only: bool
    success: bool
    error: Optional[str] = None


# --------------------------------------------------------------------------
# 2. Traditional tools — real implementations
# --------------------------------------------------------------------------

def run_sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def run_denoise_cv2(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)


def run_histogram_equalization(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def run_none(image: np.ndarray) -> np.ndarray:
    return image


# --------------------------------------------------------------------------
# 3. Deep-learning tools — STUBS
# --------------------------------------------------------------------------
# Each stub is honest about being a placeholder. Where a cheap OpenCV
# approximation makes sense for testing the pipeline's control flow, it's
# used — but it is NOT the real model's quality and shouldn't be treated
# as such by Step 6's validation loop (stub_only=True tells it that).

def run_real_esrgan_stub(image: np.ndarray) -> np.ndarray:
    """Placeholder for Real-ESRGAN. Real model does learned 4x super
    resolution; this stub does a basic 2x cv2 upscale + mild sharpen so
    there's *some* visible size/detail change to validate against."""
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return run_sharpen(upscaled)


# --------------------------------------------------------------------------
# Real Real-ESRGAN — runs via subprocess into the separate .venv-gan
# environment (Python 3.13, CUDA PyTorch, basicsr/realesrgan installed).
# Falls back to the stub automatically if the worker is unavailable,
# times out, or fails for any reason — see run_real_esrgan() below.
# --------------------------------------------------------------------------

import subprocess
import tempfile
import os as _os

REAL_ESRGAN_VENV_PYTHON = _os.path.join(".venv-gan", "Scripts", "python.exe")
REAL_ESRGAN_WORKER_SCRIPT = "real_esrgan_worker.py"
REAL_ESRGAN_TIMEOUT_SEC = 300  # generous — fp32 inference on a 4GB card can take 2+ minutes/frame

_real_esrgan_fallback_used = False  # set by run_real_esrgan(); read by tool_execution_node for accurate stub reporting


def run_real_esrgan(image: np.ndarray) -> np.ndarray:
    """Tries the real Real-ESRGAN model via subprocess into .venv-gan.
    Falls back to the stub (with a printed warning) if the .venv-gan
    environment isn't set up, the worker script is missing, inference
    times out, or anything else goes wrong — the pipeline never crashes
    because a GPU model call failed."""
    global _real_esrgan_fallback_used

    if not _os.path.isfile(REAL_ESRGAN_VENV_PYTHON) or not _os.path.isfile(REAL_ESRGAN_WORKER_SCRIPT):
        print("[real_esrgan] .venv-gan or worker script not found — using stub. "
              "See real_esrgan_worker.py setup instructions to enable the real model.")
        _real_esrgan_fallback_used = True
        return run_real_esrgan_stub(image)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = _os.path.join(tmpdir, "input.png")
        output_path = _os.path.join(tmpdir, "output.png")
        cv2.imwrite(input_path, image)

        try:
            result = subprocess.run(
                [REAL_ESRGAN_VENV_PYTHON, REAL_ESRGAN_WORKER_SCRIPT, input_path, output_path],
                capture_output=True, text=True, timeout=REAL_ESRGAN_TIMEOUT_SEC,
            )
            if result.returncode != 0:
                print(f"[real_esrgan] Worker failed (exit {result.returncode}): "
                      f"{result.stderr.strip()[-300:]} — using stub.")
                _real_esrgan_fallback_used = True
                return run_real_esrgan_stub(image)

            output = cv2.imread(output_path)
            if output is None:
                print("[real_esrgan] Worker succeeded but output file unreadable — using stub.")
                _real_esrgan_fallback_used = True
                return run_real_esrgan_stub(image)

            _real_esrgan_fallback_used = False
            return output

        except subprocess.TimeoutExpired:
            print(f"[real_esrgan] Worker timed out after {REAL_ESRGAN_TIMEOUT_SEC}s — using stub.")
            _real_esrgan_fallback_used = True
            return run_real_esrgan_stub(image)
        except Exception as exc:
            print(f"[real_esrgan] Unexpected error ({exc}) — using stub.")
            _real_esrgan_fallback_used = True
            return run_real_esrgan_stub(image)


def run_deblurgan_v2_stub(image: np.ndarray) -> np.ndarray:
    """Placeholder for DeblurGAN-v2. Real model does learned motion-blur
    removal; this stub applies an unsharp mask as a crude approximation."""
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def run_restormer_stub(image: np.ndarray) -> np.ndarray:
    """Placeholder for Restormer. Real model does learned general
    restoration; this stub chains denoise + mild contrast fix."""
    denoised = run_denoise_cv2(image)
    return run_histogram_equalization(denoised)


def run_swinir_stub(image: np.ndarray) -> np.ndarray:
    """Placeholder for SwinIR. Real model does learned restoration/SR;
    this stub does a smaller upscale than the real_esrgan stub to keep
    them visually distinguishable during testing."""
    return cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)


# --------------------------------------------------------------------------
# 4. Tool registry — maps agent's chosen action -> executor function
# --------------------------------------------------------------------------

TOOL_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sharpen": run_sharpen,
    "denoise_cv2": run_denoise_cv2,
    "histogram_equalization": run_histogram_equalization,
    "none": run_none,
    "real_esrgan": run_real_esrgan,
    "deblurgan_v2": run_deblurgan_v2_stub,
    "restormer": run_restormer_stub,
    "swinir": run_swinir_stub,
}


# --------------------------------------------------------------------------
# 5. Executor
# --------------------------------------------------------------------------

class ToolExecutor:
    def apply(self, image: np.ndarray, tool_name: str) -> ExecutionResult:
        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            return ExecutionResult(
                frame_index=-1, tool_applied=tool_name, stub_only=False,
                success=False, error=f"Unknown tool '{tool_name}' — not in registry.",
            )
        return fn  # caller wraps execution with try/except (see node below)


# --------------------------------------------------------------------------
# 6. LangGraph node function
# --------------------------------------------------------------------------

def tool_execution_node(state: PipelineStateWithFrames) -> PipelineStateWithFrames:
    """Fifth node in the graph. Wire it in as:

        graph.add_node("tool_execution", tool_execution_node)
        graph.add_edge("react_agent", "tool_execution")
        graph.add_edge("tool_execution", "validation")   # Step 6

    Reads state["enhancement_plan"] (from Step 4) and applies each
    decision's chosen tool to the matching frame's image IN PLACE
    (mutates frame.image, appends to frame.enhancement_history). Frames
    with action="none" are left untouched.
    """
    if state.get("error"):
        return state

    frames: List[FrameRecord] = state.get("frames", [])
    plan = state.get("enhancement_plan", [])
    plan_by_index = {d["frame_index"]: d for d in plan}

    results: List[ExecutionResult] = []
    frames_by_index = {f.frame_index: f for f in frames}

    for frame_index, decision in plan_by_index.items():
        frame = frames_by_index.get(frame_index)
        if frame is None:
            continue

        tool_name = decision.get("action", "none")
        fn = TOOL_FUNCTIONS.get(tool_name)

        if fn is None:
            results.append(ExecutionResult(
                frame_index=frame_index, tool_applied=tool_name,
                stub_only=False, success=False,
                error=f"Unknown tool '{tool_name}' — not in registry, frame left unchanged.",
            ))
            continue

        is_stub = tool_name in STUB_TOOLS
        try:
            frame.image = fn(frame.image)
            # real_esrgan's stub/real status is determined dynamically per call
            # (see run_real_esrgan) rather than statically, since it depends on
            # whether the .venv-gan worker succeeded this specific time.
            if tool_name == "real_esrgan":
                is_stub = _real_esrgan_fallback_used
            history_tag = f"{tool_name}[STUB]" if is_stub else tool_name
            frame.enhancement_history.append(history_tag)
            results.append(ExecutionResult(
                frame_index=frame_index, tool_applied=tool_name,
                stub_only=is_stub, success=True,
            ))
        except Exception as exc:
            results.append(ExecutionResult(
                frame_index=frame_index, tool_applied=tool_name,
                stub_only=is_stub, success=False, error=str(exc),
            ))

    return {
        **state,
        "frames": frames,
        "execution_results": [asdict(r) for r in results],
        "error": None,
    }


# --------------------------------------------------------------------------
# 7. Standalone smoke test — chains Step 1 -> 2 -> 3 -> 4 -> 5
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from step1_video_input import video_input_node
    from step2_frame_extraction import frame_extraction_node
    from step3_quality_assessment import quality_assessment_node, LocalVLMClient
    from step4_react_agent import react_agent_node, ReActAgent

    if len(sys.argv) < 2:
        print("Usage: python step5_tool_execution.py <path_to_video>")
        sys.exit(1)

    state: PipelineStateWithFrames = {"video_path": sys.argv[1], "sampling_fps": 1.0}
    state = video_input_node(state)
    if state.get("validation_status") != "valid":
        print("Step 1 failed:", state.get("error"))
        sys.exit(1)

    state = frame_extraction_node(state)
    print(f"Sampled {state['frames_sampled']} frames.")

    state = quality_assessment_node(state, vlm_client=LocalVLMClient())
    print("VLM assessment done.")

    state = react_agent_node(state, agent=ReActAgent())
    print("ReAct decisions made.")

    print("Executing tools...")
    state = tool_execution_node(state)

    for r in state["execution_results"]:
        stub_marker = " [STUB]" if r["stub_only"] else ""
        status = "OK" if r["success"] else f"FAILED: {r['error']}"
        print(f"  frame {r['frame_index']:4d}  tool={r['tool_applied']:20s}{stub_marker}  {status}")

    print("\nPer-frame enhancement history:")
    for f in state["frames"]:
        print(f"  frame {f.frame_index:4d}  history={f.enhancement_history}  shape={f.image.shape}")
