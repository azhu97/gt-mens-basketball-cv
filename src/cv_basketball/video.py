"""Thin OpenCV helpers for reading and writing video."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def video_info(path: Path) -> VideoInfo:
    cap = _open(path)
    try:
        return VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: Path) -> Iterator[tuple[int, Frame]]:
    """Yield ``(frame_idx, bgr_frame)``; indices match ultralytics' streaming order."""
    cap = _open(path)
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, np.asarray(frame, dtype=np.uint8)
            idx += 1
    finally:
        cap.release()


def read_frame(path: Path, frame_idx: int) -> Frame:
    cap = _open(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_idx} from {path}")
        return np.asarray(frame, dtype=np.uint8)
    finally:
        cap.release()


def open_writer(path: Path, info: VideoInfo) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, info.fps, (info.width, info.height))
