# node_b_vlm_worker.py
import io
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from services.qwen_vlm_service import QwenService
from schemas import DefectDetail, VLMResult
from schemas import DefectDetail, VLMResult, AnomalyDetail, AnomalyResult

app = FastAPI(title="Node B - VLM Worker")

_vlm_service: Optional[QwenService] = None


@app.on_event("startup")
async def load_model():
    global _vlm_service
    _vlm_service = QwenService()


@app.post("/analyze", response_model=VLMResult)
async def analyze(image: UploadFile = File(...), frame_number: int = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    analysis = _vlm_service.analyze_batch([pil_image])[0]

    defects = {
        category: DefectDetail(**detail)
        for category, detail in analysis.get("defects", {}).items()
    }

    return VLMResult(
        frame_number=frame_number,
        text_description=analysis.get("text_description", ""),
        defects=defects,
        overall_quality=analysis.get("overall_quality", "Unknown"),
        summary=analysis.get("summary", ""),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _vlm_service is not None}

# --- Append to node_b_vlm_worker.py (add "from schemas import AnomalyDetail, AnomalyResult" to imports) ---

@app.post("/detect_anomaly", response_model=AnomalyResult)
async def detect_anomaly(image: UploadFile = File(...), frame_number: int = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    analysis = _vlm_service.analyze_anomaly_frame(pil_image)

    anomalies = {
        category: AnomalyDetail(**detail)
        for category, detail in analysis.get("anomalies", {}).items()
    }
    return AnomalyResult(
        frame_number=frame_number,
        anomalies=anomalies,
        overall_risk=analysis.get("overall_risk", "Unknown"),
        summary=analysis.get("summary", ""),
        review_recommended=analysis.get("review_recommended", False),
    )