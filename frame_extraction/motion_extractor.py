import cv2
import os

from base_extractor import BaseFrameExtractor


class MotionExtractor(BaseFrameExtractor):

    def __init__(self,
                 video_path,
                 output_folder,
                 min_contour_area=300):

        super().__init__(video_path, output_folder)

        self.min_contour_area = min_contour_area

    def extract(self):

        os.makedirs(self.output_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)

        previous_gray = None

        saved = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            if previous_gray is None:

                previous_gray = gray

                continue

            diff = cv2.absdiff(
                previous_gray,
                gray
            )

            _, thresh = cv2.threshold(
                diff,
                15,
                255,
                cv2.THRESH_BINARY
            )

            thresh = cv2.dilate(
                thresh,
                None,
                iterations=2
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            motion = False

            for contour in contours:

                if cv2.contourArea(contour) > self.min_contour_area:

                    motion = True
                    break

            if motion:

                filename = os.path.join(
                    self.output_folder,
                    f"motion_{saved:04d}.jpg"
                )

                cv2.imwrite(filename, frame)

                saved += 1

            previous_gray = gray.copy()

        cap.release()

        print(f"Motion Extraction Complete")
        print(f"Saved {saved} frames")