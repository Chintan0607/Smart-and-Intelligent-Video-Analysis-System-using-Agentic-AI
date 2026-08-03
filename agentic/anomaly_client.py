"""
agentic/anomaly_client.py

Calls Node B's new /detect_anomaly endpoint. Kept separate from
_call_node_b (which hits /analyze) since it's a different response
model (AnomalyResult, not VLMResult) — same HTTP pattern though.
"""

import os
import httpx

from config import NODE_B_URL, HTTP_TIMEOUT_SECONDS
from schemas import AnomalyResult, ExtractedFrame


async def call_node_b_anomaly(client: httpx.AsyncClient, frame: ExtractedFrame) -> AnomalyResult:
    with open(frame.filepath, "rb") as f:
        files = {"image": (os.path.basename(frame.filepath), f.read(), "image/jpeg")}
    response = await client.post(
        f"{NODE_B_URL}/detect_anomaly",
        files=files,
        data={"frame_number": str(frame.frame_number)},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return AnomalyResult(**response.json())