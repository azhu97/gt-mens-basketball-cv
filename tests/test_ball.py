import pandas as pd
import pytest

from cv_basketball import schema as s
from cv_basketball.ball import interpolate_ball, select_ball


def _ball(frame: int, x: float, conf: float = 0.9) -> dict[str, object]:
    return {
        s.FRAME: frame,
        s.TRACK_ID: -1,
        s.LABEL: s.BALL,
        s.CONF: conf,
        "x1": x,
        "y1": 0.0,
        "x2": x + 10,
        "y2": 10.0,
    }


def test_select_ball_keeps_highest_conf_per_frame() -> None:
    balls = pd.DataFrame([_ball(0, 0, 0.2), _ball(0, 50, 0.8), _ball(1, 5, 0.5)])
    out = select_ball(balls)
    assert out[s.FRAME].tolist() == [0, 1]
    assert out["x1"].tolist() == [50, 5]


def test_interpolate_fills_short_gap_linearly() -> None:
    out = interpolate_ball(pd.DataFrame([_ball(0, 0), _ball(4, 40)]), max_gap=5)
    assert out[s.FRAME].tolist() == [0, 1, 2, 3, 4]
    assert out["x1"].tolist() == pytest.approx([0, 10, 20, 30, 40])
    assert out[s.INTERPOLATED].tolist() == [False, True, True, True, False]
    assert (out[s.LABEL] == s.BALL).all()


def test_interpolate_leaves_long_gap_empty() -> None:
    out = interpolate_ball(pd.DataFrame([_ball(0, 0), _ball(20, 200)]), max_gap=5)
    assert out[s.FRAME].tolist() == [0, 20]


def test_interpolate_empty() -> None:
    out = interpolate_ball(pd.DataFrame(columns=[s.FRAME, s.CONF, *s.BOX]))
    assert out.empty
    assert s.INTERPOLATED in out
