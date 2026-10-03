import pandas as pd

from cv_basketball import schema as s
from cv_basketball.smoothing import drop_blips, smooth_teams


def _pieces(*pieces: tuple[int, int, int, int]) -> pd.DataFrame:
    """One tracker ID cut into pieces of (track_id, team, rows, vote) in frame order."""
    rows, frame = [], 0
    for track_id, team, n, vote in pieces:
        for _ in range(n):
            rows.append(
                {
                    s.FRAME: frame,
                    s.TRACKER_ID: 1,
                    s.TRACK_ID: track_id,
                    s.TEAM: team,
                    s.TEAM_VOTE: vote,
                }
            )
            frame += 1
    return pd.DataFrame(rows)


def test_grey_piece_between_same_team_takes_it() -> None:
    # blur made 40 frames look referee-like (vote 2) in the middle of a team-0 player
    players = _pieces((1, 0, 100, 0), (2, -1, 40, 2), (3, 0, 100, 0))
    assert (smooth_teams(players) == 0).all()


def test_grey_piece_whose_votes_favour_the_other_team_stays_grey() -> None:
    players = _pieces((1, 0, 100, 0), (2, -1, 40, 1), (3, 0, 100, 0))
    assert (smooth_teams(players)[players[s.TRACK_ID] == 2] == -1).all()


def test_short_other_team_piece_is_relabelled_but_long_one_is_kept() -> None:
    short = _pieces((1, 0, 100, 0), (2, 1, 30, 1), (3, 0, 100, 0))
    assert (smooth_teams(short, max_flip_rows=45) == 0).all()
    long = _pieces((1, 0, 100, 0), (2, 1, 60, 1), (3, 0, 100, 0))
    assert (smooth_teams(long, max_flip_rows=45)[long[s.TRACK_ID] == 2] == 1).all()


def test_grey_piece_at_the_start_needs_votes_for_the_team() -> None:
    supported = _pieces((1, -1, 30, 0), (2, 0, 100, 0))
    assert (smooth_teams(supported) == 0).all()
    unsupported = _pieces((1, -1, 30, 2), (2, 0, 100, 0))
    assert (smooth_teams(unsupported)[unsupported[s.TRACK_ID] == 1] == -1).all()


def test_drop_blips_drops_short_unknown_tracks_only() -> None:
    players = _pieces((1, -1, 5, -1), (2, 0, 5, 0), (3, -1, 50, -1))
    keep = drop_blips(players, min_rows=10)
    assert set(players.loc[keep, s.TRACK_ID]) == {2, 3}
