"""Court geometry in metres and a top-down court renderer.

Coordinate system: origin at one baseline/sideline corner, x along the court length
(0..length), y along the width (0..width), as seen from the main broadcast side.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CourtSpec:
    name: str
    length: float
    width: float
    lane_width: float
    free_throw_dist: float  # baseline to free-throw line
    center_circle_radius: float

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named points used for manual homography calibration."""
        L, W = self.length, self.width
        cy, hl, ft = W / 2, self.lane_width / 2, self.free_throw_dist
        return {
            "left baseline / near sideline": (0.0, W),
            "left baseline / far sideline": (0.0, 0.0),
            "right baseline / near sideline": (L, W),
            "right baseline / far sideline": (L, 0.0),
            "halfcourt / near sideline": (L / 2, W),
            "halfcourt / far sideline": (L / 2, 0.0),
            "center court": (L / 2, cy),
            "left lane / baseline near": (0.0, cy + hl),
            "left lane / baseline far": (0.0, cy - hl),
            "left free-throw line near": (ft, cy + hl),
            "left free-throw line far": (ft, cy - hl),
            "right lane / baseline near": (L, cy + hl),
            "right lane / baseline far": (L, cy - hl),
            "right free-throw line near": (L - ft, cy + hl),
            "right free-throw line far": (L - ft, cy - hl),
        }


NBA = CourtSpec("nba", 28.65, 15.24, 4.88, 5.79, 1.83)
FIBA = CourtSpec("fiba", 28.0, 15.0, 4.9, 5.8, 1.8)
COURTS = {c.name: c for c in (NBA, FIBA)}


def draw_court(spec: CourtSpec, px_per_m: float = 10.0, margin: int = 10) -> NDArray[np.uint8]:
    """Render an empty top-down court (BGR)."""
    w = int(spec.length * px_per_m) + 2 * margin
    h = int(spec.width * px_per_m) + 2 * margin
    img = np.full((h, w, 3), (60, 110, 170), dtype=np.uint8)
    line = (255, 255, 255)

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(x * px_per_m) + margin, int(y * px_per_m) + margin

    L, W = spec.length, spec.width
    cy, hl, ft = W / 2, spec.lane_width / 2, spec.free_throw_dist
    cv2.rectangle(img, pt(0, 0), pt(L, W), line, 1)
    cv2.line(img, pt(L / 2, 0), pt(L / 2, W), line, 1)
    cv2.circle(img, pt(L / 2, cy), int(spec.center_circle_radius * px_per_m), line, 1)
    cv2.rectangle(img, pt(0, cy - hl), pt(ft, cy + hl), line, 1)
    cv2.rectangle(img, pt(L - ft, cy - hl), pt(L, cy + hl), line, 1)
    return img


def court_to_pixel(
    xy: NDArray[np.float64], px_per_m: float = 10.0, margin: int = 10
) -> NDArray[np.int32]:
    """Map court metres to pixel coords of an image made by ``draw_court``."""
    return (xy * px_per_m + margin).astype(np.int32)
