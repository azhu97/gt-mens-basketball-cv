"""Image-to-court homography from a manual calibration file."""

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
    """Corresponding image pixels and court metres, plus which court spec they refer to.

    Assumes a static camera: one calibration applies to every frame of the video.
    """

    court: str
    image_points: list[tuple[float, float]]
    court_points: list[tuple[float, float]]

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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "court": self.court,
                    "image_points": self.image_points,
                    "court_points": self.court_points,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text())
        return cls(
            court=data["court"],
            image_points=[tuple(p) for p in data["image_points"]],
            court_points=[tuple(p) for p in data["court_points"]],
        )


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


def add_court_coords(tracks: pd.DataFrame, H: NDArray[np.float64]) -> pd.DataFrame:
    court_xy = project(H, ground_points(tracks))
    return tracks.assign(**{s.COURT[0]: court_xy[:, 0], s.COURT[1]: court_xy[:, 1]})
