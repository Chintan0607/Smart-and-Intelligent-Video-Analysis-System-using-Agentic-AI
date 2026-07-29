import cv2
import os

from base_extractor import BaseFrameExtractor


class IntervalExtractor(BaseFrameExtractor):

    def __init__(self,
                 video_path,
                 output_folder,
                 interval_seconds=1):

        super().__init__(video_path, output_folder)

        self.interval_seconds = interval_seconds

    def extract(self):

        os.makedirs(self.output_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_interval = int(fps * self.interval_seconds)

        frame_count = 0
        saved = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_count % frame_interval == 0:

                filename = os.path.join(
                    self.output_folder,
                    f"frame_{saved:04d}.jpg"
                )

                cv2.imwrite(filename, frame)

                saved += 1

            frame_count += 1

        cap.release()

        print(f"Interval Extraction Complete")
        print(f"Saved {saved} frames")