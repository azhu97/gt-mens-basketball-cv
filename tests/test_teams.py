import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.teams import UNKNOWN_TEAM, assign_teams, occluded_torsos, torso_color_feature

RED, BLUE = (0, 0, 220), (220, 0, 0)


def _player_rows(
    track_id: int, bgr: tuple[int, int, int], n: int, start: int = 0
) -> list[dict[str, float]]:
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    frame[:] = bgr
    color = torso_color_feature(frame, np.array([0, 0, 100, 200], dtype=np.float32))
    return [
        {s.FRAME: f, s.TRACK_ID: track_id, **dict(zip(s.COLOR_FEATURES, color, strict=True))}
        for f in range(start, start + n)
    ]


def test_torso_feature_empty_crop_is_nan() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    feat = torso_color_feature(frame, np.array([200, 200, 210, 210], dtype=np.float32))
    assert np.isnan(feat).all()


def test_assign_teams_separates_colors_and_is_constant_per_track() -> None:
    rows = (
        _player_rows(1, RED, 5)
        + _player_rows(2, RED, 5)
        + _player_rows(3, BLUE, 5)
        + _player_rows(4, BLUE, 5)
    )
    out = assign_teams(pd.DataFrame(rows), next_track_id=5)

    assert set(out[s.TRACK_ID]) == {1, 2, 3, 4}
    per_track = out.groupby(s.TRACK_ID)[s.TEAM].nunique()
    assert (per_track == 1).all()
    by_track = out.groupby(s.TRACK_ID)[s.TEAM].first()
    assert by_track[1] == by_track[2]
    assert by_track[3] == by_track[4]
    assert by_track[1] != by_track[3]


def test_assign_teams_too_few_samples() -> None:
    players = pd.DataFrame(_player_rows(1, (0, 0, 200), 1))
    assert (assign_teams(players, next_track_id=2)[s.TEAM] == UNKNOWN_TEAM).all()


def test_id_swap_splits_track_and_keeps_early_frames_on_their_team() -> None:
    # Track 1 follows a red player for 100 frames, then a blue one for 200 (an ID swap).
    # A whole-track vote would call all 300 frames blue.
    players = pd.DataFrame(
        _player_rows(1, RED, 100)
        + _player_rows(1, BLUE, 200, start=100)
        + _player_rows(2, RED, 300)
        + _player_rows(3, BLUE, 300)
    )
    out = assign_teams(players, next_track_id=10)

    swapped = players[s.TRACK_ID] == 1
    assert (out.loc[swapped & (players[s.FRAME] < 100), s.TRACK_ID] == 1).all()
    assert (out.loc[swapped & (players[s.FRAME] >= 100), s.TRACK_ID] == 10).all()
    team = out.groupby(s.TRACK_ID)[s.TEAM].first()
    assert team[1] == team[2] != team[10] == team[3]


def test_brief_color_flicker_does_not_split() -> None:
    players = pd.DataFrame(
        _player_rows(1, RED, 100)
        + _player_rows(1, BLUE, 10, start=100)
        + _player_rows(1, RED, 100, start=110)
        + _player_rows(2, BLUE, 50)
    )
    out = assign_teams(players, next_track_id=3)
    assert set(out[s.TRACK_ID]) == {1, 2}
    assert out.groupby(s.TRACK_ID)[s.TEAM].nunique().eq(1).all()


def test_occluded_torsos_flags_overlapping_people_only() -> None:
    people = pd.DataFrame(
        {
            s.FRAME: [0, 0, 0],
            # boxes 1 and 2 overlap heavily; box 3 stands alone
            "x1": [0.0, 20.0, 300.0],
            "y1": [0.0, 10.0, 0.0],
            "x2": [100.0, 120.0, 400.0],
            "y2": [200.0, 210.0, 200.0],
        }
    )
    assert occluded_torsos(people).tolist() == [True, True, False]


def test_occluded_rows_do_not_vote_or_split() -> None:
    # Track 1 is red, but for 40 frames it overlaps blue track 2 and samples blue.
    rows = _player_rows(1, RED, 100) + _player_rows(1, BLUE, 40, start=100)
    rows += _player_rows(1, RED, 100, start=140) + _player_rows(2, BLUE, 240)
    players = pd.DataFrame(rows)
    occluded = (players[s.TRACK_ID] == 1) & players[s.FRAME].between(100, 139)

    out = assign_teams(players, next_track_id=3, occluded=occluded)
    assert set(out[s.TRACK_ID]) == {1, 2}
    team = out.groupby(s.TRACK_ID)[s.TEAM].first()
    assert team[1] != team[2]
