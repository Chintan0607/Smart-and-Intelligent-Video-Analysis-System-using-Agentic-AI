# node_c_gan_worker.py
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from enhancement_service import EnhancementService

app = FastAPI(title="Node C - GAN Worker")

_TEMP_DIR = "temp_incoming"
os.makedirs(_TEMP_DIR, exist_ok=True)

_gan_service: Optional[EnhancementService] = None


@app.on_event("startup")
async def load_model():
    global _gan_service
    _gan_service = EnhancementService()


@app.post("/enhance")
async def enhance(image: UploadFile = File(...), filename: str = Form(...)):
    temp_path = os.path.join(_TEMP_DIR, f"{uuid.uuid4()}_{filename}.jpg")
    with open(temp_path, "wb") as f:
        f.write(await image.read())

    # Unchanged heavy inference call — existing method writes its own
    # output file (see enhancement_service.py's hardcoded output path).
    _gan_service.enhance_frame(temp_path, filename)

    output_path = f"enhanced_frames/{filename}_enhanced.jpg"
    return FileResponse(output_path, media_type="image/jpeg")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _gan_service is not None}