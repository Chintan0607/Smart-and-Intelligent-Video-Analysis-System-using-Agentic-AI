import cv2
import os

from blur_metric import BlurMetric

blur = BlurMetric(threshold=100)

folder = "D:\\OpenCV\\Anomaly_Detection\\Data\\extracted_frames\\motion"

for file in sorted(os.listdir(folder)):

    if file.endswith(".jpg"):

        image_path = os.path.join(folder, file)

        image = cv2.imread(image_path)

        result = blur.evaluate(image)

        print(
            f"{file} --> Score: {result['score']} | Status: {result['status']}"
        )