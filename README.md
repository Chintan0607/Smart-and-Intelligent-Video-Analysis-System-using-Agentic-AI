# Smart and Intelligent Video Analysis System using Agentic AI

A distributed multi-service pipeline that ingests CCTV footage and produces
structured natural-language analysis of what happens in it, using an agentic
perception-action loop over a vision-language model.

**Group 5 — PGCP-AI, C-DAC ACTS Pune**

Guide: Mr. Amit Raj

Varad Kulkarni · Pratik Sonkusare · Chintan Solanke · Vivek Gotecha · Dheeraj Ippakayal

---

## Problem

CCTV footage is long, mostly redundant, and often degraded. Running a
vision-language model over every frame is computationally infeasible and largely
wasted, since consecutive frames are near-identical. And when a frame *is* worth
analysing, it may be too blurred, dark, or noisy for the model to read reliably.

The system therefore has to decide which frames are worth looking at, judge
whether each one is legible, repair it if not, and re-examine it — iterating
until the analysis is trustworthy or the attempt budget runs out.

## Architecture

Three FastAPI services, deployable across separate machines, plus a Streamlit
client.

### Node A — Agentic Orchestrator (`core/agentic_orchestrator.py`)

Runs the perception-action loop, one frame at a time:

1. Send the frame to Node B for analysis.
2. Node B returns a description *plus* a structured defect report and a
   `regeneration_recommended` flag.
3. If the frame is judged legible, stop. Otherwise `decide_tool()` selects a
   repair tool from the reported defects, applies it, and returns to step 1.
4. Capped at `MAX_AGENT_ITERATIONS` (3) with each tool attempted at most once,
   so the loop always terminates.

`decide_tool()` is a **deterministic priority router**, not an LLM-driven agent:
it matches defect keys and evidence text against keyword sets and dispatches
lighting problems to CLAHE and blur/noise/resolution problems to the GAN
enhancer. This is a design choice — the policy is auditable, has no token cost,
and adds no latency inside a loop that already runs a VLM per iteration. The
*agentic* property here is the closed loop with tool use and a convergence
check, not the mechanism of the policy itself.

**Keyframe extraction** (`core/key_frame_extractor.py`) runs first, using a
ResNet-18 embedding with cosine-similarity novelty detection at a 0.90
threshold — a frame is kept only if it differs sufficiently from the last one
kept, discarding visual redundancy before any expensive model runs.

### Node B — Hybrid VLM Worker (`nodes/node_b_vlm_worker.py`)

Vision-language analysis with a two-tier execution strategy:

- **Remote:** Qwen3.6-35B served via a vLLM OpenAI-compatible endpoint on a
  PARAM Rudra GPU node.
- **Local fallback:** Qwen3-VL-2B-Instruct with 4-bit NF4 quantization.

`HybridVLMService` falls back automatically when the remote endpoint is
unreachable, times out, or errors, so the pipeline degrades gracefully to a
smaller local model rather than failing. Configurable via `USE_REMOTE_VLLM`,
`REMOTE_VLLM_URL`, and `REMOTE_VLLM_MODEL_NAME`.

### Node C — Enhancement Service (`nodes/node_c_gan_worker.py`)

Real-ESRGAN (RRDBNet) restoration for degraded frames.

Six restoration models across GAN and transformer families — Real-ESRGAN,
SwinIR, BSRGAN, NAFNet, DeblurGAN-v2, DANet — were benchmarked on the same
degraded CCTV inputs to select the backbone. Real-ESRGAN handled the broadest
range of degradations but fell short on blur, because these models train on
synthetic degradation profiles that don't match real surveillance footage. It
was therefore fine-tuned in a fully supervised way on ~210K real degraded/clean
frame pairs, then re-benchmarked through the same harness to verify the
improvement. **Node C loads that fine-tuned checkpoint** (`weights/net_g_latest.pth`).

Benchmark harness: [GANN3](https://github.com/Pratiksonkusare/GANN3) ·
Fine-tuning: [GAN4](https://github.com/Pratiksonkusare/GAN4) ·
Module: [`realesrgan_finetune/`](realesrgan_finetune/)
— *Pratik Sonkusare*

Lightweight OpenCV corrections (CLAHE, unsharp mask, Laplacian sharpen,
histogram equalisation) run in-process on Node A for defects that don't warrant
the GAN.

### Output

Per-frame results are aggregated into a surveillance summary and rendered to a
PDF report (`services/pdf_report_generator.py`), retrievable from the job
endpoint.

## Running

Start each service in its own terminal:

```bash
uvicorn core.agentic_orchestrator:app --port 8000    # Node A
uvicorn nodes.node_b_vlm_worker:app   --port 8001    # Node B
uvicorn nodes.node_c_gan_worker:app   --port 8002    # Node C
streamlit run ui/streamlit_client.py                 # UI
```

Node addressing, iteration limits, model names, and the remote vLLM endpoint are
all set in `config.py`.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/agentic/upload` | Submit a video, returns a job id |
| `GET` | `/agentic/status/{job_id}` | Poll job progress |
| `GET` | `/agentic/report/{job_id}/pdf` | Download the PDF report |

## Repository layout

| Path | Purpose |
|---|---|
| `core/agentic_orchestrator.py` | Node A — agentic loop, job management, API |
| `core/key_frame_extractor.py` | ResNet-18 cosine-similarity keyframe selection |
| `core/opencv_tools.py` | In-process CLAHE / sharpening corrections |
| `core/video_processor.py` | Video decoding and frame handling |
| `nodes/node_b_vlm_worker.py` | Node B service wrapper |
| `nodes/node_c_gan_worker.py` | Node C service wrapper |
| `services/vllm_service.py` | Remote vLLM client with local fallback |
| `services/qwen_vlm_service.py` | Local quantized Qwen VLM |
| `services/enhancement_service.py` | Real-ESRGAN inference |
| `services/pdf_report_generator.py` | Report rendering |
| `schemas.py` | Pydantic contracts shared across services |
| `prompts.py` | VLM prompt templates |
| `realesrgan_finetune/` | Node C enhancement fine-tuning (data pipeline, configs, docs) |
| `scripts/` | Defect generation and GPU test utilities |

## Requirements

```bash
pip install -r requirements.txt
```

`requirements.lock.txt` pins exact versions for reproducing the evaluation
environment. Node B and Node C both require CUDA.

## License

MIT
