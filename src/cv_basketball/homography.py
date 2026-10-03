"""Image-to-court homography from manually clicked keyframes.

A calibration file holds one or more keyframes, each pairing image pixels in one frame
with court metres. With a single keyframe and no camera motion the camera is assumed
static; ``camera.court_homographies`` carries keyframes to every frame of a moving one.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s
from cv_basketball.court import COURTS, CourtSpec

MIN_POINTS = 4


@dataclass
class Calibration:
    """Corresponding image pixels and court metres in one frame, plus the court spec."""

    court: str
    image_points: list[tuple[float, float]]
    court_points: list[tuple[float, float]]
    frame: int = 0

    @property
    def spec(self) -> CourtSpec:
        return COURTS[self.court]

    def homography(self) -> NDArray[np.float64]:
        if len(self.image_points) < MIN_POINTS:
            raise ValueError(f"Need at least {MIN_POINTS} points, got {len(self.image_points)}")
        src = np.asarray(self.image_points, dtype=np.float64)
        dst = np.asarray(self.court_points, dtype=np.float64)
        method = cv2.RANSAC if len(src) > MIN_POINTS else 0
        H, _ = cv2.findHomography(src, dst, method)
        if H is None:
            raise ValueError("Homography estimation failed (degenerate / collinear points?)")
        return np.asarray(H, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "image_points": self.image_points,
            "court_points": self.court_points,
        }


def save_calibrations(path: Path, keyframes: list[Calibration]) -> None:
    """Write keyframes (all for the same court spec) to a JSON file."""
    if len({k.court for k in keyframes}) != 1:
        raise ValueError("All keyframes must use the same court spec")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"court": keyframes[0].court, "keyframes": [k.to_dict() for k in keyframes]}
    path.write_text(json.dumps(data, indent=2))


def load_calibrations(path: Path) -> list[Calibration]:
    """Read keyframes, sorted by frame. A file without ``keyframes`` is one at frame 0."""
    data = json.loads(path.read_text())
    entries = data.get("keyframes", [data])
    keyframes = [
        Calibration(
            court=data["court"],
            image_points=[(float(x), float(y)) for x, y in e["image_points"]],
            court_points=[(float(x), float(y)) for x, y in e["court_points"]],
            frame=int(e.get("frame", 0)),
        )
        for e in entries
    ]
    return sorted(keyframes, key=lambda k: k.frame)


def project(H: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply homography ``H`` to an (N, 2) array of points."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    return np.asarray(cv2.perspectiveTransform(pts, H), dtype=np.float64).reshape(-1, 2)


def ground_points(tracks: pd.DataFrame) -> NDArray[np.float64]:
    """Bottom-centre of each box: a player's feet on the floor.

    For the ball this is only correct when it is on the floor; airborne balls project
    past their true position.
    """
    x = ((tracks["x1"] + tracks["x2"]) / 2).to_numpy(dtype=np.float64)
    y = tracks["y2"].to_numpy(dtype=np.float64)
    return np.stack([x, y], axis=1)


def add_court_coords(
    tracks: pd.DataFrame, image_to_court: NDArray[np.float64] | dict[int, NDArray[np.float64]]
) -> pd.DataFrame:
    """Project each row's ground point to court metres.

    ``image_to_court`` is one homography for a static camera, or one per frame index;
    rows in frames without one get NaN.
    """
    if isinstance(image_to_court, np.ndarray):
        court_xy = project(image_to_court, ground_points(tracks))
    else:
        court_xy = np.full((len(tracks), 2), np.nan)
        frames = tracks[s.FRAME].to_numpy()
        ground = ground_points(tracks)
        for frame in np.unique(frames):
            if int(frame) in image_to_court:
                rows = frames == frame
                court_xy[rows] = project(image_to_court[int(frame)], ground[rows])
    return tracks.assign(**{s.COURT[0]: court_xy[:, 0], s.COURT[1]: court_xy[:, 1]})
