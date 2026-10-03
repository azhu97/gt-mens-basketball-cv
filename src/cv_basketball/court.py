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
    rim_dist: float  # baseline to rim centre
    three_pt_radius: float  # rim centre to the three-point arc
    three_pt_corner: float  # rim centre line to the straight corner three-point lines

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


NBA = CourtSpec("nba", 28.65, 15.24, 4.88, 5.79, 1.83, 1.6, 7.24, 6.71)
FIBA = CourtSpec("fiba", 28.0, 15.0, 4.9, 5.8, 1.8, 1.575, 6.75, 6.6)
# NCAA men's: NBA-sized court, 12 ft lane, three-point line at 22 ft 1.75 in since 2019-20.
NCAA = CourtSpec("ncaa", 28.65, 15.24, 3.66, 5.79, 1.83, 1.6, 6.75, 6.6)
COURTS = {c.name: c for c in (NBA, FIBA, NCAA)}


def court_lines(spec: CourtSpec, step: float = 0.1) -> list[NDArray[np.float64]]:
    """The court's painted lines as polylines in metres, sampled every ``step`` metres."""

    def seg(a: tuple[float, float], b: tuple[float, float]) -> NDArray[np.float64]:
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / step), 1)
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        return np.asarray(a) * (1 - t) + np.asarray(b) * t

    def arc(cx: float, cy: float, r: float, a0: float, a1: float) -> NDArray[np.float64]:
        a = np.linspace(a0, a1, max(int(abs(a1 - a0) * r / step), 2))
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)

    L, W = spec.length, spec.width
    cy, hl, ft = W / 2, spec.lane_width / 2, spec.free_throw_dist
    lines = [
        seg((0, 0), (L, 0)),
        seg((0, W), (L, W)),
        seg((0, 0), (0, W)),
        seg((L, 0), (L, W)),
        seg((L / 2, 0), (L / 2, W)),
        arc(L / 2, cy, spec.center_circle_radius, 0, 2 * np.pi),
    ]
    corner_x = spec.rim_dist + np.sqrt(spec.three_pt_radius**2 - spec.three_pt_corner**2)
    half_angle = float(np.arcsin(spec.three_pt_corner / spec.three_pt_radius))
    for mirrored in (False, True):
        side = [
            seg((0, cy - hl), (ft, cy - hl)),
            seg((0, cy + hl), (ft, cy + hl)),
            seg((ft, cy - hl), (ft, cy + hl)),
            arc(ft, cy, spec.center_circle_radius, -np.pi / 2, np.pi / 2),  # free-throw circle
            seg((0, cy - spec.three_pt_corner), (corner_x, cy - spec.three_pt_corner)),
            seg((0, cy + spec.three_pt_corner), (corner_x, cy + spec.three_pt_corner)),
            arc(spec.rim_dist, cy, spec.three_pt_radius, -half_angle, half_angle),
        ]
        if mirrored:
            side = [np.stack([L - p[:, 0], p[:, 1]], axis=1) for p in side]
        lines += side
    return lines


def draw_court(spec: CourtSpec, px_per_m: float = 10.0, margin: int = 10) -> NDArray[np.uint8]:
    """Render an empty top-down court (BGR)."""
    w = int(spec.length * px_per_m) + 2 * margin
    h = int(spec.width * px_per_m) + 2 * margin
    img = np.full((h, w, 3), (60, 110, 170), dtype=np.uint8)
    line_color = (255, 255, 255)
    for line in court_lines(spec):
        cv2.polylines(img, [court_to_pixel(line, px_per_m, margin)], False, line_color, 1)
    return img


def court_to_pixel(
    xy: NDArray[np.float64], px_per_m: float = 10.0, margin: int = 10
) -> NDArray[np.int32]:
    """Map court metres to pixel coords of an image made by ``draw_court``."""
    return (xy * px_per_m + margin).astype(np.int32)
