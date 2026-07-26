import cv2
import numpy as np


def _add_timestamp(frame, frame_index, fps):
    seconds = frame_index / fps
    timestamp = f"CAM 04 - 2026-07-01 23:15:{int(seconds):02d}"
    cv2.putText(
        frame,
        timestamp,
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def _add_compression_blocks(frame, frame_index):
    if frame_index % 12 != 0:
        return frame

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def create_degraded_test_video(
    filename="degraded_cctv_test.webm",
    duration_seconds=30,
    fps=15,
    width=640,
    height=360,
):
    total_frames = duration_seconds * fps
    fourcc = cv2.VideoWriter_fourcc(*("VP80" if filename.lower().endswith(".webm") else "mp4v"))
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Unable to create video file. Check OpenCV video codec support.")

    rng = np.random.default_rng(42)

    for frame_index in range(total_frames):
        frame = np.full((height, width, 3), 185, dtype=np.uint8)

        road_y = int(height * 0.68)
        cv2.rectangle(frame, (0, road_y), (width, height), (75, 75, 75), -1)
        cv2.line(frame, (0, road_y), (width, road_y), (45, 45, 45), 3)

        car_x = int(-180 + (width + 360) * frame_index / total_frames)
        car_y = road_y - 80
        cv2.rectangle(frame, (car_x, car_y), (car_x + 170, car_y + 58), (70, 40, 140), -1)
        cv2.rectangle(frame, (car_x + 34, car_y - 34), (car_x + 126, car_y), (95, 65, 170), -1)
        cv2.circle(frame, (car_x + 38, car_y + 62), 18, (20, 20, 20), -1)
        cv2.circle(frame, (car_x + 132, car_y + 62), 18, (20, 20, 20), -1)
        cv2.putText(
            frame,
            "VEHICLE-XYZ",
            (max(car_x + 8, 8), car_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        person_x = int(width * 0.78 - 90 * np.sin(frame_index / 20))
        person_y = road_y - 110
        cv2.circle(frame, (person_x, person_y), 18, (35, 35, 35), -1)
        cv2.line(frame, (person_x, person_y + 18), (person_x, person_y + 78), (35, 35, 35), 6)
        cv2.line(frame, (person_x, person_y + 45), (person_x - 28, person_y + 72), (35, 35, 35), 5)
        cv2.line(frame, (person_x, person_y + 45), (person_x + 28, person_y + 72), (35, 35, 35), 5)

        _add_timestamp(frame, frame_index, fps)

        noise = rng.normal(0, 22, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (9, 9), 0)
        frame = _add_compression_blocks(frame, frame_index)

        writer.write(frame)

    writer.release()
    print(f"Success! Test video saved as: {filename}")
    print(f"Duration: {duration_seconds} seconds")
    print("You can now upload this video file to your Streamlit app.")


if __name__ == "__main__":
    create_degraded_test_video()
