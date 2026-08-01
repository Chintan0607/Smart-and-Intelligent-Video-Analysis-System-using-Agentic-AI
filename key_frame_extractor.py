# key_frame_extractor.py
from __future__ import annotations

import os
from typing import AsyncGenerator, Optional, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from config import MODEL_CACHE_DIR
from schemas import ExtractedFrame, ExtractionMethod, FrameMetadata


class ShotBoundaryDetector(Protocol):
    def predict_frame(self, frame_bgr: np.ndarray) -> float: ...


class NoOpShotBoundaryDetector:
    def predict_frame(self, frame_bgr: np.ndarray) -> float:
        return 0.0


class _ResNetEmbedder:
    """
    Loads ResNet-18 from models/huggingface/. If the local checkpoint is
    missing, downloads torchvision's default pretrained weights once,
    then caches them locally so subsequent runs never hit the network.
    """

    def __init__(self, device: str = "cuda", local_weights_name: str = "resnet18_backbone.pth"):
        self.device = device if torch.cuda.is_available() else "cpu"

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        weights_path = os.path.join(MODEL_CACHE_DIR, local_weights_name)

        backbone = models.resnet18(weights=None)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            backbone.load_state_dict(state_dict)
        else:
            pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone.load_state_dict(pretrained.state_dict())
            torch.save(backbone.state_dict(), weights_path)

        backbone.fc = torch.nn.Identity()
        self.model = backbone.to(self.device).eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed(self, frame_rgb: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(frame_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return F.normalize(features.squeeze(0), dim=0)


class KeyframeExtractor:
    def __init__(
        self,
        output_folder: str,
        similarity_threshold: float = 0.90,
        candidate_stride: int = 5,
        max_gap_frames: int = 150,
        shot_boundary_detector: Optional[ShotBoundaryDetector] = None,
        device: str = "cuda",
    ):
        self.output_folder = output_folder
        self.similarity_threshold = similarity_threshold
        self.candidate_stride = candidate_stride
        self.max_gap_frames = max_gap_frames
        self.shot_detector = shot_boundary_detector or NoOpShotBoundaryDetector()
        self.embedder = _ResNetEmbedder(device=device)
        os.makedirs(self.output_folder, exist_ok=True)

    async def extract(self, video_path: str) -> AsyncGenerator[ExtractedFrame, None]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_number = 0
        last_kept_number = -self.max_gap_frames
        last_embedding: Optional[torch.Tensor] = None
        current_shot_id = 0

        while True:
            success = cap.grab()
            if not success:
                break

            is_candidate = frame_number % self.candidate_stride == 0
            gap_exceeded = (frame_number - last_kept_number) >= self.max_gap_frames

            if is_candidate or gap_exceeded:
                ok, frame_bgr = cap.retrieve()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                boundary_score = self.shot_detector.predict_frame(frame_bgr)
                is_shot_boundary = boundary_score >= 0.5
                if is_shot_boundary:
                    current_shot_id += 1
                    last_embedding = None

                embedding = self.embedder.embed(frame_rgb)

                similarity = (
                    float(torch.dot(embedding, last_embedding))
                    if last_embedding is not None
                    else 0.0
                )
                is_novel = last_embedding is None or similarity < self.similarity_threshold

                should_keep = is_shot_boundary or is_novel or gap_exceeded
                if not should_keep:
                    frame_number += 1
                    continue

                if is_shot_boundary:
                    method = ExtractionMethod.SHOT_BOUNDARY
                    confidence = boundary_score
                elif gap_exceeded and not is_novel:
                    method = ExtractionMethod.FORCED_INTERVAL
                    confidence = 0.5
                else:
                    method = ExtractionMethod.SEMANTIC_NOVELTY
                    confidence = round(1.0 - similarity, 4)

                filename = f"frame_{frame_number:06d}.jpg"
                filepath = os.path.join(self.output_folder, filename)
                cv2.imwrite(filepath, frame_bgr)

                metadata = FrameMetadata(
                    frame_number=frame_number,
                    timestamp_sec=round(frame_number / fps, 3),
                    shot_id=current_shot_id,
                    extraction_method=method,
                    confidence=confidence,
                    similarity_to_previous=round(similarity, 4) if last_embedding is not None else None,
                )

                yield ExtractedFrame(frame_number=frame_number, filepath=filepath, metadata=metadata)

                last_embedding = embedding
                last_kept_number = frame_number

            frame_number += 1

        cap.release()