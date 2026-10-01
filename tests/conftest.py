from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """A 10-frame 320x240 video with a moving colored rectangle (no real players)."""
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (20 + 10 * i, 60), (60 + 10 * i, 180), (0, 0, 200), -1)
        writer.write(frame)
    writer.release()
    return path
