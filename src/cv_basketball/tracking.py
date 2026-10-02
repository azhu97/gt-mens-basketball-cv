"""YOLO player tracking (ultralytics BoT-SORT / ByteTrack) plus per-frame ball detection."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cv_basketball.video import iter_frames

# COCO class indices used by the pretrained YOLO weights.
PERSON = 0
SPORTS_BALL = 32

NO_TRACK = -1

DEFAULT_TRACKER = str(Path(__file__).with_name("botsort_basketball.yaml"))


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
    tracker: str = DEFAULT_TRACKER,
    person_conf: float = 0.4,
    ball_conf: float = 0.1,
    ball_imgsz: int = 1280,
) -> Iterator[FrameDetections]:
    """Stream detections frame by frame without holding the video in memory.

    Players go through ``model.track``. The ball is found with a separate ``predict``
    at a larger ``ball_imgsz``: at the default 640 a 1080p ball shrinks to ~10 px, and
    the tracker would also drop its short-lived detections. Ball rows get ``NO_TRACK``.
    Confidences are detector floors; callers may apply stricter thresholds.
    """
    from ultralytics import YOLO  # lazy: heavy import

    # Separate instances: predict() on the tracking model would run its tracker callbacks.
    player_model = YOLO(model_name)
    ball_model = YOLO(model_name)
    for idx, image in iter_frames(video):
        p = player_model.track(
            image,
            persist=True,
            classes=[PERSON],
            device=device,
            tracker=tracker,
            conf=person_conf,
            verbose=False,
        )[0].boxes
        b = ball_model.predict(
            image,
            classes=[SPORTS_BALL],
            device=device,
            conf=ball_conf,
            imgsz=ball_imgsz,
            verbose=False,
        )[0].boxes
        player_ids = (
            p.id.cpu().numpy().astype(np.int64)
            if p.id is not None
            else np.full(len(p), NO_TRACK, dtype=np.int64)
        )
        yield FrameDetections(
            frame_idx=idx,
            image=image,
            xyxy=np.concatenate([p.xyxy.cpu().numpy(), b.xyxy.cpu().numpy()]),
            track_ids=np.concatenate([player_ids, np.full(len(b), NO_TRACK, dtype=np.int64)]),
            classes=np.concatenate(
                [p.cls.cpu().numpy().astype(np.int64), b.cls.cpu().numpy().astype(np.int64)]
            ),
            confs=np.concatenate([p.conf.cpu().numpy(), b.conf.cpu().numpy()]),
        )
