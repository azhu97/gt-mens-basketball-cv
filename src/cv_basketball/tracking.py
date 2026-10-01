"""YOLO detection + ultralytics built-in multi-object tracking (ByteTrack / BoT-SORT)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# COCO class indices used by the pretrained YOLO weights.
PERSON = 0
SPORTS_BALL = 32

NO_TRACK = -1


@dataclass(frozen=True)
class FrameDetections:
    frame_idx: int
    image: NDArray[np.uint8]  # original BGR frame
    xyxy: NDArray[np.float32]  # (N, 4) pixel boxes
    track_ids: NDArray[np.int64]  # (N,), NO_TRACK where the tracker gave no ID
    classes: NDArray[np.int64]  # (N,) COCO class indices
    confs: NDArray[np.float32]  # (N,)


def track_video(
    video: Path,
    *,
    model_name: str,
    device: str,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.1,
) -> Iterator[FrameDetections]:
    """Stream tracked detections frame by frame without holding the video in memory.

    ``conf`` is the detector floor; callers apply stricter per-class thresholds.
    Note: ``model.track`` only returns boxes the tracker has confirmed, so short-lived
    detections (often the ball) can be dropped. See ``ball.interpolate_ball``.
    """
    from ultralytics import YOLO  # lazy: heavy import

    model = YOLO(model_name)
    results = model.track(
        source=str(video),
        stream=True,
        persist=True,
        classes=[PERSON, SPORTS_BALL],
        device=device,
        tracker=tracker,
        conf=conf,
        verbose=False,
    )
    for idx, r in enumerate(results):
        boxes = r.boxes
        n = len(boxes)
        ids = (
            boxes.id.cpu().numpy().astype(np.int64)
            if boxes.id is not None
            else np.full(n, NO_TRACK, dtype=np.int64)
        )
        yield FrameDetections(
            frame_idx=idx,
            image=r.orig_img,
            xyxy=boxes.xyxy.cpu().numpy(),
            track_ids=ids,
            classes=boxes.cls.cpu().numpy().astype(np.int64),
            confs=boxes.conf.cpu().numpy(),
        )
