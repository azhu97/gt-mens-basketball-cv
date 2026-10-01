from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cv_basketball import schema as s
from cv_basketball.court import NBA
from cv_basketball.homography import Calibration, add_court_coords, project


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


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cal = _scaled_calibration()
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = Calibration.load(path)
    assert loaded == cal
    assert loaded.spec is NBA


def test_add_court_coords_uses_box_bottom_center() -> None:
    H = _scaled_calibration().homography()
    tracks = pd.DataFrame({"x1": [80.0], "y1": [0.0], "x2": [120.0], "y2": [60.0]})
    out = add_court_coords(tracks, H)
    assert out[s.COURT].to_numpy()[0] == pytest.approx([5.0, 3.0], abs=1e-6)
