from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cv_basketball import schema as s
from cv_basketball.court import NBA
from cv_basketball.homography import (
    Calibration,
    add_court_coords,
    load_calibrations,
    project,
    reject_bad_keyframes,
    save_calibrations,
)


def _scaled_calibration() -> Calibration:
    # Image is the court at 20 px/m, so the true homography is a uniform scale.
    lm = NBA.landmarks()
    names = [
        "left baseline / far sideline",
        "right baseline / far sideline",
        "right baseline / near sideline",
        "left baseline / near sideline",
        "center court",
    ]
    court = [lm[n] for n in names]
    image = [(x * 20, y * 20) for x, y in court]
    return Calibration(court="nba", image_points=image, court_points=court)


def test_homography_recovers_scale() -> None:
    H = _scaled_calibration().homography()
    out = project(H, np.array([[100.0, 60.0], [0.0, 0.0]]))
    assert out == pytest.approx(np.array([[5.0, 3.0], [0.0, 0.0]]), abs=1e-6)


def test_too_few_points() -> None:
    cal = Calibration("nba", [(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
    with pytest.raises(ValueError):
        cal.homography()


def test_save_load_roundtrip_sorts_keyframes(tmp_path: Path) -> None:
    late, early = _scaled_calibration(), _scaled_calibration()
    late.frame = 90
    path = tmp_path / "cal.json"
    save_calibrations(path, [late, early])
    loaded = load_calibrations(path)
    assert loaded == [early, late]
    assert loaded[0].spec is NBA


def test_load_reads_old_single_calibration_format(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    path.write_text('{"court": "nba", "image_points": [[0, 0]], "court_points": [[1, 2]]}')
    (cal,) = load_calibrations(path)
    assert cal.frame == 0
    assert cal.court_points == [(1.0, 2.0)]


def test_add_court_coords_uses_box_bottom_center() -> None:
    H = _scaled_calibration().homography()
    tracks = pd.DataFrame({"x1": [80.0], "y1": [0.0], "x2": [120.0], "y2": [60.0]})
    out = add_court_coords(tracks, H)
    assert out[s.COURT].to_numpy()[0] == pytest.approx([5.0, 3.0], abs=1e-6)


def test_add_court_coords_per_frame() -> None:
    H = _scaled_calibration().homography()
    box = {"x1": 80.0, "y1": 0.0, "x2": 120.0, "y2": 60.0}
    tracks = pd.DataFrame([{s.FRAME: 0, **box}, {s.FRAME: 1, **box}])
    out = add_court_coords(tracks, {0: H, 1: 2 * np.eye(3) @ H})  # scaled H is the same map
    assert out[s.COURT].to_numpy() == pytest.approx(np.array([[5.0, 3.0], [5.0, 3.0]]), abs=1e-6)
    assert np.isnan(add_court_coords(tracks, {0: H})[s.COURT[0]].iloc[1])


def test_mis_clicked_keyframe_is_rejected() -> None:
    # 100 px per metre: pixel (100x, 100y) is court (x, y)
    good = Calibration(
        "nba",
        [(0, 0), (2865, 0), (0, 1524), (2865, 1524)],
        [(0, 0), (28.65, 0), (0, 15.24), (28.65, 15.24)],
        frame=0,
    )
    # the same clicks as if they were a quarter of the court: everything lands 4x out
    bad = Calibration(
        "nba",
        [(0, 0), (716, 0), (0, 381), (716, 381)],
        [(0, 0), (28.65, 0), (0, 15.24), (28.65, 15.24)],
        frame=1,
    )
    feet = [(500.0, 300.0), (1500.0, 800.0), (2000.0, 1200.0), (1000.0, 500.0)]
    people = pd.DataFrame(
        [
            {s.FRAME: f, "x1": x - 20, "y1": y - 150, "x2": x + 20, "y2": y, s.FLOOR_DIST: 1.0}
            for f in (0, 1)
            for x, y in feet
        ]
    )
    kept, rejected = reject_bad_keyframes([good, bad], people)
    assert kept == [good]
    assert rejected == [bad]
