import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.ball2d import NO_OWNER, ground_ball


def _rows() -> pd.DataFrame:
    """Player 1 holds the ball for frames 0-4, passes (5-9, ball in the air), player 2
    holds it for 10-14 and keeps it; after frame 14 nobody does."""
    rows = []
    for f in range(18):
        for tid, x, court_x in ((1, 100.0, 2.0), (2, 600.0, 5.0)):
            rows.append(
                {
                    s.FRAME: f,
                    s.TRACK_ID: tid,
                    s.LABEL: s.PLAYER,
                    "x1": x,
                    "y1": 100.0,
                    "x2": x + 60,
                    "y2": 300.0,
                    s.COURT[0]: court_x,
                    s.COURT[1]: 5.0,
                }
            )
        if f < 5:
            bx = 130.0
        elif f < 10:
            bx = 130.0 + (f - 4) * 90  # in flight, high above the floor
        elif f < 15:
            bx = 630.0
        else:
            bx = 900.0
        # in flight the floor projection is far off; held, it's wherever
        rows.append(
            {
                s.FRAME: f,
                s.TRACK_ID: -1,
                s.LABEL: s.BALL,
                "x1": bx - 5,
                "y1": 195.0,
                "x2": bx + 5,
                "y2": 205.0,
                s.COURT[0]: 40.0,
                s.COURT[1]: -9.0,
            }
        )
    return pd.DataFrame(rows)


def test_held_ball_is_at_the_holders_feet_and_flies_in_a_line() -> None:
    out = ground_ball(_rows())
    ball = out[out[s.LABEL] == s.BALL].set_index(s.FRAME)
    assert (ball.loc[0:4, s.OWNER] == 1).all() and (ball.loc[10:14, s.OWNER] == 2).all()
    assert np.allclose(ball.loc[0:4, s.COURT[0]], 2.0)
    assert np.allclose(ball.loc[10:14, s.COURT[0]], 5.0)
    # in flight: evenly from 2 m to 5 m over frames 4..10 (15 m/s)
    assert np.allclose(ball.loc[5:9, s.COURT[0]], 2.0 + np.arange(1, 6) * 3 / 6)
    assert np.allclose(ball.loc[5:9, s.COURT[1]], 5.0)
    # after the last hold the ball can't be placed
    assert ball.loc[15:, s.COURT[0]].isna().all()
    assert (out.loc[out[s.LABEL] == s.PLAYER, s.OWNER] == NO_OWNER).all()


def test_ball_passing_a_player_for_one_frame_is_not_held() -> None:
    rows = _rows()
    ball = rows[s.LABEL] == s.BALL
    # frame 7: the pass crosses player 2's box for a single frame
    rows.loc[ball & (rows[s.FRAME] == 7), ["x1", "x2"]] = [620.0, 630.0]
    out = ground_ball(rows)
    assert out.loc[(out[s.LABEL] == s.BALL) & (out[s.FRAME] == 7), s.OWNER].item() == NO_OWNER


def test_hold_that_needs_an_impossibly_fast_pass_is_dropped() -> None:
    rows = _rows()
    # player 2 is 30 m away: reaching them in 6 frames would be a 150 m/s pass, so
    # their (shorter-or-equal) hold is the wrong one and the ball is never placed there
    rows.loc[(rows[s.TRACK_ID] == 2), s.COURT[0]] = 32.0
    out = ground_ball(rows)
    ball = out[out[s.LABEL] == s.BALL]
    assert (ball[s.OWNER] != 2).all()
    assert not np.isclose(ball[s.COURT[0]], 32.0).any()
