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


def test_sixth_player_with_least_support_moves_to_the_short_team() -> None:
    from cv_basketball.smoothing import grey_out_sixth_players

    rows = []
    for f in range(20):
        # five solid team-0 players, four team-1 players, and track 9: tagged team 0
        # but its own recent votes say team 1
        for tid, team, vote in [
            *((i, 0, 0) for i in range(5)),
            *((i, 1, 1) for i in range(5, 9)),
            (9, 0, 1),
        ]:
            rows.append({s.FRAME: f, s.TRACK_ID: tid, s.TEAM: team, s.TEAM_VOTE: vote})
    players = pd.DataFrame(rows)
    out = grey_out_sixth_players(players)
    assert (out[players[s.TRACK_ID] == 9] == 1).all()
    assert (out[players[s.TRACK_ID] != 9] == players.loc[players[s.TRACK_ID] != 9, s.TEAM]).all()


def _court(roster: list[tuple[int, int, int]], frames: int = 20) -> pd.DataFrame:
    """Rows of (track_id, team, vote) for every frame."""
    return pd.DataFrame(
        [
            {s.FRAME: f, s.TRACK_ID: tid, s.TEAM: team, s.TEAM_VOTE: vote}
            for f in range(frames)
            for tid, team, vote in roster
        ]
    )


def test_grey_player_joins_the_team_that_is_one_short() -> None:
    from cv_basketball.smoothing import fill_to_five

    # five on team 0, four on team 1, and a hidden player with no clean votes
    roster = [(i, 0, 0) for i in range(5)] + [(i, 1, 1) for i in range(5, 9)] + [(9, -1, -1)]
    players = _court(roster)
    assert (fill_to_five(players)[players[s.TRACK_ID] == 9] == 1).all()


def test_grey_player_stays_grey_when_both_teams_are_full_or_it_looks_like_a_ref() -> None:
    from cv_basketball.smoothing import fill_to_five

    full = [(i, 0, 0) for i in range(5)] + [(i, 1, 1) for i in range(5, 10)]
    eleventh = _court([*full, (10, -1, -1)])
    assert (fill_to_five(eleventh)[eleventh[s.TRACK_ID] == 10] == -1).all()
    # one short, but its votes say referee (2)
    ref = _court([*full[:9], (10, -1, 2)])
    assert (fill_to_five(ref)[ref[s.TRACK_ID] == 10] == -1).all()


def test_two_extras_are_moved_one_at_a_time() -> None:
    from cv_basketball.smoothing import grey_out_sixth_players

    # team 0 shows 7: five solid, plus track 9 (votes say team 1) and track 10 (no
    # clean votes). Team 1 has 4, so it has room for one more but not two.
    roster = (
        [(i, 0, 0) for i in range(5)] + [(i, 1, 1) for i in range(5, 9)] + [(9, 0, 1), (10, 0, -1)]
    )
    players = _court(roster)
    out = grey_out_sixth_players(players)
    assert (out[players[s.TRACK_ID] == 9] == 1).all()  # the hidden opponent moves over
    assert (out[players[s.TRACK_ID] == 10] == -1).all()  # no room left: grey
