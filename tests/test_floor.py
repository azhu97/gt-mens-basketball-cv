import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.floor import floor_distance, floor_hull, on_court_tracks


def _arena() -> np.ndarray:
    """Dark stands on top, a bright floor below with a dark painted logo in the middle."""
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame[200:, :] = (150, 190, 220)
    frame[300:360, 250:400] = 40
    return frame


def test_floor_hull_covers_floor_and_logo_but_not_stands() -> None:
    hull = floor_hull(_arena())
    assert hull is not None
    box_h = 100.0

    def dist(foot_x: float, foot_y: float) -> float:
        return floor_distance(hull, np.array([foot_x - 20, foot_y - box_h, foot_x + 20, foot_y]))

    assert dist(100, 400) > 0  # on the floor
    assert dist(320, 330) > 0  # on the dark logo, filled in by the hull
    assert dist(320, 120) < -0.5  # in the stands, almost a box height above the floor


def test_floor_distance_without_hull_is_nan() -> None:
    assert np.isnan(floor_distance(None, np.array([0, 0, 10, 10], dtype=np.float32)))


def test_on_court_tracks_votes_per_track() -> None:
    people = pd.DataFrame(
        {
            # fan: always off; player: steps onto a dark sideline band for 2 of 10 frames
            s.TRACK_ID: [1] * 10 + [2] * 10 + [-1, -1],
            s.FLOOR_DIST: [-1.2] * 10 + [0.3] * 8 + [-0.9] * 2 + [-1.0, np.nan],
        }
    )
    keep = on_court_tracks(people)
    assert not keep[people[s.TRACK_ID] == 1].any()
    assert keep[people[s.TRACK_ID] == 2].all()
    assert keep.iloc[-2:].tolist() == [False, True]
