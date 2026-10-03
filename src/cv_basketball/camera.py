"""Broadcast camera motion, and per-frame image-to-court homographies from keyframes.

A broadcast camera pans, tilts and zooms from a fixed spot, so consecutive frames are
related by a homography over the whole image (crowd and floor alike). Pass 1 measures it
with sparse optical flow on everything except people. ``court_homographies`` then
carries each hand-calibrated keyframe to every other frame through those motions. Flow
drifts slowly, so each frame blends the predictions of the keyframes before and after it,
weighted by how near they are in time.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from cv_basketball.court import COURTS
from cv_basketball.homography import Calibration

_SCALE = 0.5  # measure motion on a half-size frame: faster, and plenty accurate
_MAX_CORNERS = 600
_MIN_POINTS = 12
_GRID = 30  # court grid resolution used to blend homographies

Homography = NDArray[np.float64]


def frame_motion(
    prev_gray: NDArray[np.uint8], gray: NDArray[np.uint8], people: NDArray[np.float32]
) -> Homography:
    """Homography mapping pixels of the previous frame to this one (identity on failure).

    ``people`` holds (N, 4) person boxes in this frame; they move on their own, so
    features on them are skipped.
    """
    small_prev = cv2.resize(prev_gray, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_AREA)
    small = cv2.resize(gray, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_AREA)
    mask = np.full(small.shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in np.asarray(people, dtype=np.float64) * _SCALE:
        cv2.rectangle(mask, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), 0, -1)
    p0 = cv2.goodFeaturesToTrack(small_prev, _MAX_CORNERS, 0.01, 6, mask=mask)
    if p0 is None or len(p0) < _MIN_POINTS:
        return np.eye(3)
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        small_prev, small, p0, p0.copy(), winSize=(21, 21), maxLevel=4
    )
    ok = status.ravel() == 1
    if ok.sum() < _MIN_POINTS:
        return np.eye(3)
    M, _ = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.0)
    if M is None:
        return np.eye(3)
    S = np.diag([_SCALE, _SCALE, 1.0])
    return np.asarray(np.linalg.inv(S) @ M @ S, dtype=np.float64)


def _blend(
    court_to_image: list[Homography], weights: list[float], grid: NDArray[np.float64]
) -> Homography:
    """Weighted blend of court->image homographies, through projected court grid points."""
    pts = np.zeros_like(grid)
    for H, w in zip(court_to_image, weights, strict=True):
        projected = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H)
        pts += w * np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    blended, _ = cv2.findHomography(grid, pts / sum(weights))
    return np.asarray(blended, dtype=np.float64)


def court_homographies(
    motions: NDArray[np.float64], keyframes: list[Calibration]
) -> NDArray[np.float64]:
    """Image->court homography for every frame, from per-frame motions and keyframes.

    ``motions[i]`` maps frame ``i - 1`` to frame ``i`` (``motions[0]`` is unused). A frame
    between two keyframes linearly blends their predictions; outside the keyframes it uses
    the nearest one.
    """
    if not keyframes:
        raise ValueError("Need at least one calibrated keyframe")
    n = len(motions)
    # cumulative[i] maps frame-0 pixels to frame-i pixels
    cumulative = np.empty((n, 3, 3))
    cumulative[0] = np.eye(3)
    for i in range(1, n):
        cumulative[i] = motions[i] @ cumulative[i - 1]
    spec = COURTS[keyframes[0].court]
    gx, gy = np.meshgrid(np.linspace(0, spec.length, _GRID), np.linspace(0, spec.width, _GRID // 2))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # court->frame-0 pixels, per keyframe
    anchors = {
        k.frame: np.linalg.inv(cumulative[k.frame]) @ np.linalg.inv(k.homography())
        for k in keyframes
        if k.frame < n
    }
    frames = sorted(anchors)
    out = np.empty((n, 3, 3))
    for i in range(n):
        after = next((f for f in frames if f >= i), frames[-1])
        before = next((f for f in reversed(frames) if f <= i), frames[0])
        if before == after:
            c2i = cumulative[i] @ anchors[before]
        else:
            w_after = (i - before) / (after - before)
            c2i = _blend(
                [cumulative[i] @ anchors[before], cumulative[i] @ anchors[after]],
                [1 - w_after, w_after],
                grid,
            )
        out[i] = np.linalg.inv(c2i)
    return out
