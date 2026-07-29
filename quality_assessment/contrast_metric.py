import cv2
import numpy as np

from base_metric import BaseQualityMetric


class ContrastMetric(BaseQualityMetric):

    def __init__(self, threshold=40):
        """
        threshold:
        < 40  -> Low Contrast
        >=40  -> Good Contrast
        """
        self.threshold = threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        contrast = np.std(gray)

        if contrast < self.threshold:
            status = "LOW"
        else:
            status = "GOOD"

        return {

            "metric": "Contrast",

            "score": round(float(contrast), 2),

            "status": status

        }