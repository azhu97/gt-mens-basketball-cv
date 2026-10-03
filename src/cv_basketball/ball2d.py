"""Place the ball on the court: at its holder's feet, or on a line between holders.

Projecting the ball's box onto the floor is only right when the ball is on the floor. A
ball held at chest height, or in the air on a pass or shot, projects far behind its
true position, so on the court it jumps around at impossible speeds.

Instead, a ball inside a player's box (between their head and feet, give or take a
little to the sides) is *held* by that player and placed at their feet on the court.
Between two held stretches the ball is in flight or loose, and is moved in a straight
line from the last holder's position to the next one's. A hold must last a few frames:
a ball passing in front of a player for a frame or two isn't held. Before the first
hold and after the last there is nothing to anchor the ball to, so it gets no court
position (NaN) rather than a wrong one.
"""

import numpy as np
import pandas as pd

from cv_basketball import schema as s

NO_OWNER = -1


def ball_owners(tracks: pd.DataFrame, side_margin: float = 0.2) -> "pd.Series[int]":
    """Track ID of the player holding the ball in each ball row (``NO_OWNER`` if none).

    The ball is held when its centre is inside a player's box, widened by
    ``side_margin`` box widths on each side (a ball held out at arm's length). With
    several such players, the one whose box centre is horizontally nearest wins.
    """
    balls = tracks[tracks[s.LABEL] == s.BALL]
    players = tracks[(tracks[s.LABEL] == s.PLAYER) & (tracks[s.TRACK_ID] >= 0)]
    by_frame = dict(tuple(players.groupby(s.FRAME)))
    owners = pd.Series(NO_OWNER, index=balls.index, dtype="int64")
    for pos, (_, ball) in enumerate(balls.iterrows()):
        frame = by_frame.get(ball[s.FRAME])
        if frame is None:
            continue
        cx, cy = (ball["x1"] + ball["x2"]) / 2, (ball["y1"] + ball["y2"]) / 2
        w = frame["x2"] - frame["x1"]
        inside = (
            (cx >= frame["x1"] - side_margin * w)
            & (cx <= frame["x2"] + side_margin * w)
            & (cy >= frame["y1"])
            & (cy <= frame["y2"])
        )
        if inside.any():
            near = np.abs((frame["x1"] + frame["x2"]).to_numpy() / 2 - cx)
            near[~inside.to_numpy()] = np.inf
            owners.iloc[pos] = int(frame[s.TRACK_ID].to_numpy()[np.argmin(near)])
    return owners


def _drop_brief_holds(owners: "pd.Series[int]", min_hold: int) -> "pd.Series[int]":
    """``owners`` (in frame order) with runs of one holder shorter than ``min_hold`` cleared."""
    runs = (owners != owners.shift()).cumsum()
    length = owners.groupby(runs).transform("size")
    return owners.where(length >= min_hold, NO_OWNER)


def _drop_impossible_holds(
    held: np.ndarray, owner: np.ndarray, xy: np.ndarray, frames: np.ndarray, max_speed: float
) -> np.ndarray:
    """``held`` with holds cleared until no flight between holds needs over ``max_speed``.

    Between two holds the ball flies in a line; if that line is faster than any pass
    (``max_speed`` in metres per frame), one of the two holds is wrong, typically a
    ball passing in front of a player. The shorter of the two is cleared, and the check
    repeats.
    """
    held = held.copy()
    while True:
        idx = np.flatnonzero(held)
        if len(idx) < 2:
            return held
        # runs of one holder over consecutive held rows
        starts = [0, *(np.flatnonzero(owner[idx][1:] != owner[idx][:-1]) + 1)]
        runs = [idx[a:b] for a, b in zip(starts, [*starts[1:], len(idx)], strict=True)]
        worst = None
        for r0, r1 in zip(runs, runs[1:], strict=False):
            dist = np.linalg.norm(xy[r1[0]] - xy[r0[-1]])
            if dist > max_speed * (frames[r1[0]] - frames[r0[-1]]):
                worst = r0 if len(r0) < len(r1) else r1
                break
        if worst is None:
            return held
        held[worst] = False


def ground_ball(
    tracks: pd.DataFrame, min_hold: int = 3, max_pass_speed: float = 20.0, fps: float = 30.0
) -> pd.DataFrame:
    """Return ``tracks`` with the ball's court position at its holder or between holders.

    Adds ``OWNER`` (holder's track ID on ball rows, ``NO_OWNER`` elsewhere); a holder
    must keep the ball for ``min_hold`` consecutive ball rows, and holds that would need
    a pass faster than ``max_pass_speed`` m/s between them are dropped (see
    ``_drop_impossible_holds``). Needs court coordinates.
    """
    out = tracks.copy()
    out[s.OWNER] = NO_OWNER
    balls = out[(out[s.LABEL] == s.BALL) & out[s.COURT[0]].notna()].sort_values(s.FRAME)
    if balls.empty:
        return out
    owners = ball_owners(out.loc[balls.index.union(out.index[out[s.LABEL] == s.PLAYER])])
    owners = _drop_brief_holds(owners.loc[balls.index], min_hold)
    out.loc[owners.index, s.OWNER] = owners

    players = out[out[s.LABEL] == s.PLAYER].groupby([s.FRAME, s.TRACK_ID])[s.COURT].first()
    xy = balls[s.COURT].to_numpy(dtype=np.float64).copy()
    held = np.zeros(len(balls), dtype=bool)
    for i, (frame, owner) in enumerate(zip(balls[s.FRAME], owners.loc[balls.index], strict=True)):
        if owner != NO_OWNER and (frame, owner) in players.index:
            xy[i] = players.loc[(frame, owner)].to_numpy(dtype=np.float64)
            held[i] = True

    # straight lines between held stretches (passes, shots to a rebounder, loose balls)
    frames = balls[s.FRAME].to_numpy(dtype=np.float64)
    owner_arr = owners.loc[balls.index].to_numpy()
    held = _drop_impossible_holds(held, owner_arr, xy, frames, max_pass_speed / fps)
    owners = owners.where(held, NO_OWNER)
    out.loc[owners.index, s.OWNER] = owners
    if held.any():
        first, last = np.flatnonzero(held)[[0, -1]]
        order = np.arange(len(balls))
        between = ~held & (order > first) & (order < last)
        for axis in range(2):
            xy[between, axis] = np.interp(frames[between], frames[held], xy[held, axis])
        xy[(order < first) | (order > last)] = np.nan
    else:
        xy[:] = np.nan
    out.loc[balls.index, s.COURT[0]] = xy[:, 0]
    out.loc[balls.index, s.COURT[1]] = xy[:, 1]
    return out
