import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.teams import UNKNOWN_TEAM, assign_teams, torso_color_feature


def _player_rows(track_id: int, bgr: tuple[int, int, int], n: int) -> list[dict[str, float]]:
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[:] = bgr
    color = torso_color_feature(frame, np.array([0, 0, 100, 200], dtype=np.float32))
    return [
        {s.FRAME: f, s.TRACK_ID: track_id, **dict(zip(s.COLOR_FEATURES, color, strict=True))}
        for f in range(n)
    ]


def test_torso_feature_empty_crop_is_nan() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    feat = torso_color_feature(frame, np.array([200, 200, 210, 210], dtype=np.float32))
    assert np.isnan(feat).all()


def test_assign_teams_separates_colors_and_is_constant_per_track() -> None:
    red, blue = (0, 0, 220), (220, 0, 0)
    rows = (
        _player_rows(1, red, 5)
        + _player_rows(2, red, 5)
        + _player_rows(3, blue, 5)
        + _player_rows(4, blue, 5)
    )
    players = pd.DataFrame(rows)
    teams = assign_teams(players)

    per_track = teams.groupby(players[s.TRACK_ID]).nunique()
    assert (per_track == 1).all()
    by_track = teams.groupby(players[s.TRACK_ID]).first()
    assert by_track[1] == by_track[2]
    assert by_track[3] == by_track[4]
    assert by_track[1] != by_track[3]


def test_assign_teams_too_few_samples() -> None:
    players = pd.DataFrame(_player_rows(1, (0, 0, 200), 1))
    assert (assign_teams(players) == UNKNOWN_TEAM).all()
