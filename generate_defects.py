import os
import random
import cv2
import numpy as np

# Force OpenCV FFmpeg backend to bypass V4L2 hardware driver checks
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264"


class VideoDefectGenerator:

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def apply_blur(self, frame, severity="medium"):
        size_map = {"low": 7, "medium": 15, "high": 25}
        kernel_size = size_map.get(severity, 15)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        return cv2.filter2D(frame, -1, kernel)

    def apply_noise(self, frame, severity="medium"):
        std_map = {"low": 15, "medium": 30, "high": 55}
        std = std_map.get(severity, 30)
        noise = np.random.normal(0, std, frame.shape).astype(np.float32)
        noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255)
        return noisy_frame.astype(np.uint8)

    def apply_compression(self, frame, severity="medium"):
        quality_map = {"low": 30, "medium": 12, "high": 5}
        quality = quality_map.get(severity, 12)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, result = cv2.imencode(".jpg", frame, encode_param)
        return cv2.imdecode(result, 1)

    def apply_lighting(self, frame, defect_type="underexposure"):
        if defect_type == "underexposure":
            return cv2.convertScaleAbs(frame, alpha=0.35, beta=0)
        else:
            return cv2.convertScaleAbs(frame, alpha=1.6, beta=40)

    def process_video(self, input_path, output_dir):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            return

        filename = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Use mp4v standard software encoding
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        targets = {
            "test_blur": os.path.join(output_dir, f"{filename}_defect_blur.mp4"),
            "test_noise": os.path.join(
                output_dir, f"{filename}_defect_noise.mp4"
            ),
            "test_compression": os.path.join(
                output_dir, f"{filename}_defect_compression.mp4"
            ),
            "test_lighting": os.path.join(
                output_dir, f"{filename}_defect_lighting.mp4"
            ),
            "test_random": os.path.join(
                output_dir, f"{filename}_defect_random_mixed.mp4"
            ),
        }

        writers = {
            key: cv2.VideoWriter(path, fourcc, fps, (width, height))
            for key, path in targets.items()
        }

        print(f"\nProcessing: {filename} ...")
        all_defects = ["blur", "noise", "compression", "lighting"]
        random_active_defects = random.sample(
            all_defects, k=random.randint(1, 3)
        )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            writers["test_blur"].write(self.apply_blur(frame, severity="medium"))
            writers["test_noise"].write(
                self.apply_noise(frame, severity="medium")
            )
            writers["test_compression"].write(
                self.apply_compression(frame, severity="medium")
            )
            writers["test_lighting"].write(
                self.apply_lighting(frame, defect_type="underexposure")
            )

            mixed_frame = frame.copy()
            if "blur" in random_active_defects:
                mixed_frame = self.apply_blur(mixed_frame, severity="low")
            if "noise" in random_active_defects:
                mixed_frame = self.apply_noise(mixed_frame, severity="low")
            if "compression" in random_active_defects:
                mixed_frame = self.apply_compression(
                    mixed_frame, severity="medium"
                )
            if "lighting" in random_active_defects:
                mixed_frame = self.apply_lighting(
                    mixed_frame, defect_type="underexposure"
                )

            writers["test_random"].write(mixed_frame)

        cap.release()
        for w in writers.values():
            w.release()

        print(
            f"Finished processing {frame_count} frames. Generated compact MP4 files in '{output_dir}'."
        )


if __name__ == "__main__":
    INPUT_VIDEOS_DIR = "/home/varad/ML_Workspace/CDAC_Project/test_videos/"
    OUTPUT_TEST_DIR = "/home/varad/ML_Workspace/CDAC_Project/test_videos/videos_with_defects/"

    generator = VideoDefectGenerator()

    if os.path.exists(INPUT_VIDEOS_DIR):
        # Purge the huge AVI files first to free disk space
        if os.path.exists(OUTPUT_TEST_DIR):
            for f in os.listdir(OUTPUT_TEST_DIR):
                if f.endswith(".avi"):
                    os.remove(os.path.join(OUTPUT_TEST_DIR, f))

        for file in os.listdir(INPUT_VIDEOS_DIR):
            video_path = os.path.join(INPUT_VIDEOS_DIR, file)
            if os.path.isfile(video_path) and file.endswith(
                (".mp4", ".mkv", ".avi", ".mov")
            ):
                generator.process_video(video_path, OUTPUT_TEST_DIR)