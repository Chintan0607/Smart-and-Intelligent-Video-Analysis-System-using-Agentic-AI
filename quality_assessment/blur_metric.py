import cv2

from base_metric import BaseQualityMetric


class BlurMetric(BaseQualityMetric):

    def __init__(self, threshold=100):
        """
        threshold:
            Lower value = more tolerant
            Higher value = stricter
        """
        self.threshold = threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        variance = laplacian.var()

        if variance >= self.threshold:
            status = "SHARP"
        else:
            status = "BLURRY"

        return {

            "metric": "Blur",

            "score": round(variance, 2),

            "threshold": self.threshold,

            "status": status
        }