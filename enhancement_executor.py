import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class EnhancementExecutor:
    def __init__(self, output_folder: str = "enhanced_frames"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_rgb_array(frame: Any) -> np.ndarray:
        if isinstance(frame, np.ndarray):
            arr = frame
        else:
            arr = np.asarray(frame)

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    @staticmethod
    def _apply_deblurgan_v2(frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_restormer(frame: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame, None, 12, 12, 7, 21)

    @staticmethod
    def _apply_histogram_equalization(frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(eq, cv2.COLOR_YCrCb2RGB)

    @staticmethod
    def _apply_real_esrgan(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        return cv2.resize(frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _apply_contrast_enhancement(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    def apply_enhancement(self, frame: Any, tool_name: str) -> np.ndarray:
        rgb = self._to_rgb_array(frame)

        if tool_name == "deblurgan_v2":
            return self._apply_deblurgan_v2(rgb)
        if tool_name == "restormer":
            return self._apply_restormer(rgb)
        if tool_name == "histogram_equalization":
            return self._apply_histogram_equalization(rgb)
        if tool_name == "real_esrgan":
            return self._apply_real_esrgan(rgb)
        if tool_name == "contrast_enhancement":
            return self._apply_contrast_enhancement(rgb)

        return rgb

    def save_enhanced_frame(self, frame: np.ndarray, frame_number: int) -> str:
        filename = f"enhanced_frame_{frame_number:05d}.jpg"
        path = self.output_folder / filename
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return str(path)

    def enhance_and_save_frames(
        self,
        frames_by_number: Dict[int, Any],
        selected_frame_numbers: List[int],
        tool_name: str,
    ) -> List[Dict[str, Any]]:
        enhanced_results = []
        for frame_number in selected_frame_numbers:
            frame = frames_by_number.get(frame_number)
            if frame is None:
                continue

            enhanced_frame = self.apply_enhancement(frame, tool_name)
            path = self.save_enhanced_frame(enhanced_frame, frame_number)

            enhanced_results.append(
                {
                    "frame_number": frame_number,
                    "tool": tool_name,
                    "enhanced_image": enhanced_frame,
                    "output_path": path,
                }
            )

        return enhanced_results
