import cv2
import numpy as np

from base_metric import BaseQualityMetric


class SharpnessMetric(BaseQualityMetric):

    def __init__(self, threshold=50):
        """
        threshold:
        < 50  -> Soft Image
        >=50  -> Sharp Image
        """
        self.threshold = threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        sharpness = np.mean(magnitude)

        if sharpness >= self.threshold:
            status = "SHARP"
        else:
            status = "SOFT"

        return {
            "metric": "Sharpness",
            "score": round(float(sharpness), 2),
            "status": status
        }