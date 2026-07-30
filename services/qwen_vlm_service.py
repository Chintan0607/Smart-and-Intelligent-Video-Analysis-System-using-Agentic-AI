import logging

import cv2
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MAX_NEW_TOKENS, MODEL_CACHE_DIR, VLM_MODEL_CPU

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_TEMPLATE = """
You are an expert surveillance image quality analyst.
Based on the frame metrics and detected issue flags below, write a concise summary for this frame.
Keep the summary short, factual, and mention only the detected issues.

Frame metrics:
- blur_variance: {blur_variance:.1f}
- brightness: {brightness:.1f}
- noise_std: {noise_std:.1f}
- contrast: {contrast:.1f}
- saturation: {saturation:.1f}

Detected issues:
- Blur: {blur_present}
- Noise: {noise_present}
- Compression: {compression_present}
- Lighting: {lighting_issue}
- Overall Quality: {overall_quality}
"""


class QwenService:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_error = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                VLM_MODEL_CPU,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                VLM_MODEL_CPU,
                cache_dir=MODEL_CACHE_DIR,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.model.eval()
            logger.info("Loaded CPU-friendly text model successfully.")
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            self.model = None
            self.tokenizer = None
            self._load_error = str(exc)
            logger.warning(
                "CPU-friendly model could not be loaded; using heuristic analysis. Error: %s",
                exc,
            )

    def _compute_frame_metrics(self, frame):
        image_array = np.asarray(frame)
        if image_array.ndim == 3 and image_array.shape[2] >= 3:
            rgb = image_array[..., :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.asarray(image_array, dtype=np.uint8)
            rgb = None

        gray = gray.astype(np.float32)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        brightness = float(gray.mean())
        noise_std = float(gray.std())
        contrast = float(gray.max() - gray.min())
        saturation = float(
            np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1])
            if rgb is not None
            else 0.0
        )

        return {
            "blur_variance": blur_variance,
            "brightness": brightness,
            "noise_std": noise_std,
            "contrast": contrast,
            "saturation": saturation,
        }

    def _classify_metrics(self, metrics):
        blur_present = metrics["blur_variance"] < 120.0
        noise_present = metrics["noise_std"] > 40.0
        lighting_issue = metrics["brightness"] < 80.0 or metrics["brightness"] > 220.0
        compression_present = metrics["contrast"] < 30.0 or metrics["saturation"] < 35.0

        return {
            "defects": {
                "Blur": {
                    "present": blur_present,
                    "confidence": "medium",
                    "severity": "high" if blur_present else "low",
                    "evidence": (
                        "Low edge sharpness detected"
                        if blur_present
                        else "Edge detail appears clear"
                    ),
                },
                "Noise": {
                    "present": noise_present,
                    "confidence": "medium",
                    "severity": "high" if noise_present else "low",
                    "evidence": (
                        "Image noise level is elevated"
                        if noise_present
                        else "Image noise level is acceptable"
                    ),
                },
                "Compression": {
                    "present": compression_present,
                    "confidence": "medium",
                    "severity": "medium" if compression_present else "low",
                    "evidence": (
                        "Compression artifacts may be present"
                        if compression_present
                        else "Compression artifacts are not obvious"
                    ),
                },
                "Lighting": {
                    "present": lighting_issue,
                    "confidence": "medium",
                    "severity": "medium" if lighting_issue else "low",
                    "evidence": (
                        "Lighting is unusually dark or bright"
                        if lighting_issue
                        else "Lighting appears balanced"
                    ),
                },
            },
            "overall_quality": (
                "Needs review"
                if any(
                    [
                        blur_present,
                        noise_present,
                        compression_present,
                        lighting_issue,
                    ]
                )
                else "Good"
            ),
        }

    def _build_summary_prompt(self, metrics, classification):
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            blur_variance=metrics["blur_variance"],
            brightness=metrics["brightness"],
            noise_std=metrics["noise_std"],
            contrast=metrics["contrast"],
            saturation=metrics["saturation"],
            blur_present="yes" if classification["defects"]["Blur"]["present"] else "no",
            noise_present="yes" if classification["defects"]["Noise"]["present"] else "no",
            compression_present="yes" if classification["defects"]["Compression"]["present"] else "no",
            lighting_issue="yes" if classification["defects"]["Lighting"]["present"] else "no",
            overall_quality=classification["overall_quality"],
        )
        return prompt

    def _query_vlm(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        decoded_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded_text.strip()

    def _is_valid_summary(self, text):
        if not text:
            return False
        cleaned = text.strip().lower()
        if len(cleaned.split()) < 5:
            return False
        if cleaned in {"blur", "noise", "compression", "lighting", "unknown", "yes", "no"}:
            return False
        return True

    def _build_fallback_summary(self, classification):
        issues = []
        for name, defect in classification["defects"].items():
            if defect["present"]:
                issues.append(name.lower())

        if not issues:
            return "Frame quality is good with no major issues detected."

        issue_text = ", ".join(issues)
        return f"Detected issues in this frame: {issue_text}. Review the selected frames for potential quality concerns."

    def analyze_frame(self, frame):
        metrics = self._compute_frame_metrics(frame)
        classification = self._classify_metrics(metrics)

        summary = None
        if self.model is not None and self.tokenizer is not None:
            try:
                prompt = self._build_summary_prompt(metrics, classification)
                candidate = self._query_vlm(prompt)
                if self._is_valid_summary(candidate):
                    summary = candidate
                else:
                    logger.debug("Model returned an unusable summary; falling back to deterministic text. Output: %s", candidate)
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                logger.warning(
                    "Summary generation failed; falling back to deterministic analysis. Error: %s",
                    exc,
                )

        if summary is None:
            summary = self._build_fallback_summary(classification)

        return {
            "text_description": summary,
            "defects": classification["defects"],
            "overall_quality": classification["overall_quality"],
            "summary": summary,
        }

    def _heuristic_analysis(self, frame, error=None):
        image_array = np.asarray(frame)
        if image_array.ndim == 3:
            if image_array.shape[2] >= 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array[:, :, 0]
        else:
            gray = image_array

        gray = np.asarray(gray, dtype=np.uint8)
        if gray.ndim != 2:
            gray = gray[:, :, 0]

        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            laplacian_var = float(np.var(laplacian))
        except Exception:
            gx = np.gradient(gray.astype(np.float32), axis=1)
            gy = np.gradient(gray.astype(np.float32), axis=0)
            laplacian_var = float(np.var(np.sqrt(gx * gx + gy * gy)))

        brightness = float(gray.mean())
        noise = float(gray.std())

        blur_present = laplacian_var < 120.0
        lighting_issue = brightness < 85.0 or brightness > 220.0
        compression_issue = noise < 20.0
        noise_issue = noise > 45.0

        detail = (
            f"Heuristic quality check used because the Qwen model could not be initialized. "
            f"blur_variance={laplacian_var:.1f}, brightness={brightness:.1f}, noise={noise:.1f}"
        )
        if error:
            detail += f"; model_error={error}"

        return {
            "text_description": detail,
            "defects": {
                "Blur": {
                    "present": blur_present,
                    "confidence": "medium",
                    "severity": "high" if blur_present else "low",
                    "evidence": "Low edge sharpness detected" if blur_present else "Edge detail appears clear",
                },
                "Noise": {
                    "present": noise_issue,
                    "confidence": "medium",
                    "severity": "medium" if noise_issue else "low",
                    "evidence": "Image noise level is elevated" if noise_issue else "Image noise level is acceptable",
                },
                "Compression": {
                    "present": compression_issue,
                    "confidence": "medium",
                    "severity": "medium" if compression_issue else "low",
                    "evidence": "Compression artifacts may be present" if compression_issue else "Compression artifacts are not obvious",
                },
                "Lighting": {
                    "present": lighting_issue,
                    "confidence": "medium",
                    "severity": "medium" if lighting_issue else "low",
                    "evidence": "Lighting is unusually dark or bright" if lighting_issue else "Lighting appears balanced",
                },
            },
            "overall_quality": "Needs review" if any([blur_present, noise_issue, compression_issue, lighting_issue]) else "Good",
            "summary": "Heuristic defect assessment completed because the primary VLM model was unavailable.",
        }
