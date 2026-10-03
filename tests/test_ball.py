import pandas as pd
import pytest

from cv_basketball import schema as s
from cv_basketball.ball import interpolate_ball, track_ball


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


def _person(frame: int, x: float) -> dict[str, object]:
    return {s.FRAME: frame, "x1": x - 20, "y1": 0.0, "x2": x + 20, "y2": 100.0}


NO_PEOPLE = pd.DataFrame(columns=[s.FRAME, *s.BOX])


def test_track_ball_prefers_smooth_path_over_confident_jumps() -> None:
    # The real ball moves 5 px/frame; a more confident decoy flickers between two far
    # spots in every other frame, so no real ball could link its detections.
    real = [_ball(f, 100 + 5 * f, 0.75) for f in range(20)]
    decoy = [_ball(f, 600 + 900 * (f % 4 == 0), 0.9) for f in range(0, 20, 2)]
    out = track_ball(pd.DataFrame(real + decoy), NO_PEOPLE)
    assert out[s.FRAME].tolist() == list(range(20))
    assert (out["x1"] < 300).all()


def test_track_ball_prefers_ball_near_a_player_over_static_decoy() -> None:
    # A confident decoy that never moves (the mascot) versus a less confident ball
    # that stays with a player.
    real = [_ball(f, 100 + 3 * f, 0.5) for f in range(20)]
    decoy = [_ball(f, 800, 0.85) for f in range(20)]
    people = pd.DataFrame([_person(f, 105 + 3 * f) for f in range(20)])
    out = track_ball(pd.DataFrame(real + decoy), people)
    assert (out["x1"] < 300).all()


def test_track_ball_drops_lone_weak_detection() -> None:
    out = track_ball(pd.DataFrame([_ball(0, 50, 0.3)]), NO_PEOPLE)
    assert out.empty


def test_interpolate_fills_short_gap_linearly() -> None:
    out = interpolate_ball(pd.DataFrame([_ball(0, 0), _ball(4, 40)]), max_gap=5)
    assert out[s.FRAME].tolist() == [0, 1, 2, 3, 4]
    assert out["x1"].tolist() == pytest.approx([0, 10, 20, 30, 40])
    assert out[s.INTERPOLATED].tolist() == [False, True, True, True, False]
    assert (out[s.LABEL] == s.BALL).all()


def test_interpolate_does_not_bridge_an_impossible_jump() -> None:
    # 3 missing frames can't cover 1000 px for a 10 px ball.
    out = interpolate_ball(pd.DataFrame([_ball(0, 0), _ball(4, 1000)]), max_gap=5)
    assert out[s.FRAME].tolist() == [0, 4]


def test_interpolate_leaves_long_gap_empty() -> None:
    out = interpolate_ball(pd.DataFrame([_ball(0, 0), _ball(20, 200)]), max_gap=5)
    assert out[s.FRAME].tolist() == [0, 20]


def test_interpolate_empty() -> None:
    out = interpolate_ball(pd.DataFrame(columns=[s.FRAME, s.CONF, *s.BOX]))
    assert out.empty
    assert s.INTERPOLATED in out
