"""Per-frame court floor outline, used to drop people who aren't on the court (fans, staff).

The broadcast camera pans, so a one-off calibration can't say where the court is in each
frame. The lit floor is the brightest large region of an arena frame and the stands are
dark, so an Otsu threshold on lightness finds it in every frame. Its convex hull fills in
dark painted logos and the players standing on it.
"""

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s

_SCALE = 0.25  # work on a downscaled frame; the outline only needs to be coarse
_OPEN_KERNEL = np.ones((7, 7), np.uint8)
# Keep bright regions at least this share of the largest, so a strip of floor cut off by
# dark sideline lettering still counts.
_MIN_REGION_SHARE = 0.05
# A person's feet may be this many box heights outside the bright floor and still count as
# on court: players step onto dark painted bands along the sidelines.
ON_COURT_MARGIN = 0.5
# Tracks on court in no more than this share of their frames are dropped.
MIN_ON_COURT_SHARE = 0.5


def floor_hull(frame: NDArray[np.uint8]) -> NDArray[np.int32] | None:
    """Convex hull (full-resolution pixels) of the bright court floor, or None if not found."""
    small = cv2.resize(frame, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_AREA)
    lightness = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0], (5, 5), 0)
    _, mask = cv2.threshold(lightness, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Opening cuts thin bright links (ad text, cheerleaders) between floor and stands.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    n, regions, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= _MIN_REGION_SHARE * areas.max())
    ys, xs = np.nonzero(np.isin(regions, keep))
    points = np.column_stack([xs, ys]).astype(np.int32)
    return np.asarray(cv2.convexHull(points) / _SCALE, dtype=np.int32)


def floor_distance(hull: NDArray[np.int32] | None, xyxy: NDArray[np.float32]) -> float:
    """Signed distance of a box's bottom-centre from the floor hull, in box heights.

    Positive inside, negative outside; NaN if there's no hull.
    """
    if hull is None:
        return float("nan")
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    dist = cv2.pointPolygonTest(hull, ((x1 + x2) / 2, y2), True)
    return float(dist) / max(y2 - y1, 1.0)


def on_court_tracks(people: pd.DataFrame) -> "pd.Series[bool]":
    """Flag rows of people who are on the court, deciding once per track.

    A fan in the stands is off the floor in every frame, while a player only leaves it
    briefly, so each track is kept if its feet are near the floor in more than
    ``MIN_ON_COURT_SHARE`` of its frames. Rows with no track ID are judged on their own,
    and rows with no floor distance count as on court.
    """
    dist = people[s.FLOOR_DIST]
    near = (dist > -ON_COURT_MARGIN) | dist.isna()
    tracked = people[s.TRACK_ID] >= 0
    share = near[tracked].astype(float).groupby(people.loc[tracked, s.TRACK_ID]).mean()
    keep = near.copy()
    keep.loc[tracked] = people.loc[tracked, s.TRACK_ID].map(share > MIN_ON_COURT_SHARE)
    return keep.astype(bool)
