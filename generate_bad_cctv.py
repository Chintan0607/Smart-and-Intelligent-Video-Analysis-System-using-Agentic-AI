import cv2
import numpy as np
from pathlib import Path

output = Path(__file__).resolve().parent / "bad_quality_cctv.mp4"
output.parent.mkdir(parents=True, exist_ok=True)
width, height = 320, 240
fps = 15
seconds = 12
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height), isColor=False)

for i in range(fps * seconds):
    t = i / fps
    frame = np.full((height, width), 35, dtype=np.uint8)
    noise = (np.random.randn(height, width) * 20).astype(np.int16)
    flicker = np.clip(frame.astype(np.int16) + noise + (np.sin(t * 3.0) * 25).astype(np.int16), 0, 255).astype(np.uint8)
    frame = flicker
    cv2.circle(frame, (int((width / 2) + np.sin(t * 1.2) * 70), int((height / 2) + np.cos(t * 1.5) * 25)), 16, int(80 + 40 * np.sin(t * 4.5)), -1)
    frame = cv2.GaussianBlur(frame, (9, 9), 3)
    small = cv2.resize(frame, (width // 8, height // 8), interpolation=cv2.INTER_LINEAR)
    frame = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    for y in range(0, height, 16):
        for x in range(0, width, 16):
            if np.random.rand() < 0.1:
                block = frame[y:y+16, x:x+16].astype(np.int16)
                block = np.clip(block + np.random.randint(-18, 18), 0, 255).astype(np.uint8)
                frame[y:y+16, x:x+16] = block
    writer.write(frame)

writer.release()
print(f"Created: {output}")
