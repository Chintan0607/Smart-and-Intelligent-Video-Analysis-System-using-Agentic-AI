import io
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from services.qwen_vlm_service import QwenService
from schemas import DefectDetail, VLMResult

app = FastAPI(title="Node B - VLM Worker")

_vlm_service: Optional[QwenService] = None

# Any defect name containing one of these substrings, if flagged
# `present=True` by the VLM, is grounds for another agent iteration.
# Substring matching (not exact-key matching) so variants like
# "motion_blur" or "low_lighting" still trigger correctly.
_CONCERNING_DEFECT_KEYWORDS = ["blur", "noise", "compression", "lighting"]
_CONCERNING_QUALITY_LABELS = {"fair", "poor"}


@app.on_event("startup")
async def load_model():
    global _vlm_service
    _vlm_service = QwenService()


def _compute_regeneration_recommended(defects: dict[str, DefectDetail], overall_quality: str) -> bool:
    """
    Bug fix: VLMResult.regeneration_recommended defaults to False on the
    Pydantic model, and the old endpoint never set it explicitly — so
    every frame silently reported "no regeneration needed" regardless of
    what `defects` actually contained, and the agent loop broke on
    iteration 1 every time. This computes it explicitly from the
    analysis instead of relying on the field default.
    """
    has_concerning_defect = any(
        detail.present and any(keyword in name.lower() for keyword in _CONCERNING_DEFECT_KEYWORDS)
        for name, detail in defects.items()
    )
    quality_flagged = overall_quality.strip().lower() in _CONCERNING_QUALITY_LABELS
    return has_concerning_defect or quality_flagged


@app.post("/analyze", response_model=VLMResult)
async def analyze(image: UploadFile = File(...), frame_number: int = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    analysis = _vlm_service.analyze_batch([pil_image])[0]

    defects = {
        category: DefectDetail(**detail)
        for category, detail in analysis.get("defects", {}).items()
    }
    overall_quality = analysis.get("overall_quality", "Unknown")

    return VLMResult(
        frame_number=frame_number,
        text_description=analysis.get("text_description", ""),
        defects=defects,
        overall_quality=overall_quality,
        summary=analysis.get("summary", ""),
        regeneration_recommended=_compute_regeneration_recommended(defects, overall_quality),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _vlm_service is not None}