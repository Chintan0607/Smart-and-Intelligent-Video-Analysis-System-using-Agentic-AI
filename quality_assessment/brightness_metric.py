import cv2

from base_metric import BaseQualityMetric


class BrightnessMetric(BaseQualityMetric):

    def __init__(
        self,
        dark_threshold=70,
        bright_threshold=180
    ):
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold

    def evaluate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        brightness = gray.mean()

        if brightness < self.dark_threshold:
            status = "DARK"

        elif brightness > self.bright_threshold:
            status = "BRIGHT"

        else:
            status = "NORMAL"

        return {

            "metric": "Brightness",

            "score": round(brightness, 2),

            "status": status

        }