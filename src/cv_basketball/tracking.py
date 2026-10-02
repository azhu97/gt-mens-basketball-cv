"""YOLO player tracking (ultralytics BoT-SORT / ByteTrack) plus per-frame ball detection."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cv_basketball import schema as s
from cv_basketball.video import iter_frames

# Model class names -> schema labels. Covers pretrained COCO weights ("person",
# "sports ball") and fine-tuned basketball weights (see ``dataset.TARGET_CLASSES``).
_LABELS_BY_NAME = {
    "person": s.PLAYER,
    "player": s.PLAYER,
    "referee": s.REFEREE,
    "sports ball": s.BALL,
    "ball": s.BALL,
}

NO_TRACK = -1

DEFAULT_TRACKER = str(Path(__file__).with_name("botsort_basketball.yaml"))


@dataclass(frozen=True)
class FrameDetections:
    frame_idx: int
    image: NDArray[np.uint8]  # original BGR frame
    xyxy: NDArray[np.float32]  # (N, 4) pixel boxes
    track_ids: NDArray[np.int64]  # (N,), NO_TRACK where the tracker gave no ID
    labels: tuple[str, ...]  # (N,) schema labels: s.PLAYER / s.REFEREE / s.BALL
    confs: NDArray[np.float32]  # (N,)


def label_map(names: dict[int, str]) -> dict[int, str]:
    """Map a model's class IDs to schema labels by class name; unknown classes are left out."""
    return {i: _LABELS_BY_NAME[n] for i, n in names.items() if n in _LABELS_BY_NAME}


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

    Players (and referees, if the model has that class) go through ``model.track``. The
    ball is found with a separate ``predict`` at a larger ``ball_imgsz``: at the default
    640 a 1080p ball shrinks to ~10 px, and the tracker would also drop its short-lived
    detections. Ball rows get ``NO_TRACK``.
    Confidences are detector floors; callers may apply stricter thresholds.
    """
    from ultralytics import YOLO  # lazy: heavy import

    # Separate instances: predict() on the tracking model would run its tracker callbacks.
    player_model = YOLO(model_name)
    ball_model = YOLO(model_name)
    labels = label_map(player_model.names)
    person_classes = [i for i, lab in labels.items() if lab != s.BALL]
    ball_classes = [i for i, lab in labels.items() if lab == s.BALL]
    if not person_classes or not ball_classes:
        raise ValueError(f"{model_name} has no player or ball class: {player_model.names}")
    for idx, image in iter_frames(video):
        p = player_model.track(
            image,
            persist=True,
            classes=person_classes,
            device=device,
            tracker=tracker,
            conf=person_conf,
            verbose=False,
        )[0].boxes
        b = ball_model.predict(
            image,
            classes=ball_classes,
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
            labels=tuple(labels[int(c)] for c in [*p.cls.tolist(), *b.cls.tolist()]),
            confs=np.concatenate([p.conf.cpu().numpy(), b.conf.cpu().numpy()]),
        )
