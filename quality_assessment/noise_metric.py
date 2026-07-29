import cv2
import numpy as np

from base_metric import BaseQualityMetric


class NoiseMetric(BaseQualityMetric):

    def __init__(self, threshold=10):
        """
        threshold:
        <10  -> Low Noise
        >=10 -> High Noise
        """
        self.threshold = threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Estimate noise using Laplacian
        noise = cv2.Laplacian(gray, cv2.CV_64F).std()

        if noise >= self.threshold:
            status = "HIGH"
        else:
            status = "LOW"

        return {
            "metric": "Noise",
            "score": round(float(noise), 2),
            "status": status
        }