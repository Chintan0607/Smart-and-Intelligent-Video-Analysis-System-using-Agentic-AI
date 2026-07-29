import os
import cv2

from quality_assessment.blur_metric import BlurMetric
from quality_assessment.brightness_metric import BrightnessMetric

FRAME_FOLDER = "data/extracted_frames"

blur_detector = BlurMetric(threshold=100)

brightness_detector = BrightnessMetric()

print("=" * 90)
print("VIDEO QUALITY ASSESSMENT")
print("=" * 90)

for file in sorted(os.listdir(FRAME_FOLDER)):

    if file.endswith(".jpg"):

        image_path = os.path.join(FRAME_FOLDER, file)

        image = cv2.imread(image_path)

        blur = blur_detector.evaluate(image)

        brightness = brightness_detector.evaluate(image)

        print(
            f"{file:<18}"
            f"Blur: {blur['score']:<10}"
            f"{blur['status']:<10}"
            f"Brightness: {brightness['score']:<10}"
            f"{brightness['status']}"
        )

print("=" * 90)
print("Completed")