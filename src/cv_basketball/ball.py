"""Ball post-processing: one ball per frame, chosen as a path through time, then gap filling.

Most frames have several ball candidates. Some are confident false positives that look
like a ball: the mascot's head, the rim, orange shoes. Taking the most confident
candidate per frame jumps between them. Instead, ``track_ball`` picks the sequence of
candidates that scores best overall, where consecutive picks must be close enough for a
real ball to travel between them, and candidates near a player score extra. A smooth
path alone isn't enough, because a false positive that stands still (the mascot) is
perfectly smooth.
"""

import numpy as np
import pandas as pd

from cv_basketball import schema as s

# Parameters chosen against hand-labelled frames from two clips (see learning log).
MAX_SPEED = 3.5  # ball widths per frame a real ball can move (fast pass ~2 at 30 fps)
MAX_LINK_GAP = 10  # frames the path may skip between two picks
MIN_CONF = 0.7  # a candidate scores conf - MIN_CONF, so weaker ones only bridge gaps...
NEAR_PLAYER_BONUS = 1.0  # ...unless near a player, where the ball usually is
NEAR_PLAYER_PAD = 0.5  # "near": within this share of the box size around a person
GAP_COST = 0.05  # per skipped frame on a link
RESTART_COST = 1.0  # to start the path again somewhere unreachable


def near_people(balls: pd.DataFrame, people: pd.DataFrame) -> "pd.Series[bool]":
    """Flag ball candidates whose centre is in or just around a person box in the same frame."""
    near = pd.Series(False, index=balls.index)
    by_frame = dict(tuple(people.groupby(s.FRAME)))
    for frame, cands in balls.groupby(s.FRAME):
        if frame not in by_frame:
            continue
        boxes = by_frame[frame][s.BOX].to_numpy(dtype=np.float64)
        w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        lo_x, hi_x = boxes[:, 0] - NEAR_PLAYER_PAD * w, boxes[:, 2] + NEAR_PLAYER_PAD * w
        lo_y, hi_y = boxes[:, 1] - NEAR_PLAYER_PAD * h, boxes[:, 3] + 0.1 * h  # not below feet
        cx = ((cands[s.BOX[0]] + cands[s.BOX[2]]) / 2).to_numpy()[:, None]
        cy = ((cands[s.BOX[1]] + cands[s.BOX[3]]) / 2).to_numpy()[:, None]
        inside = (cx > lo_x) & (cx < hi_x) & (cy > lo_y) & (cy < hi_y)
        near.loc[cands.index] = inside.any(axis=1)
    return near


def track_ball(balls: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Return at most one ball row per frame: the best-scoring path through the candidates.

    Dynamic programming over candidates in frame order. Each candidate's best score is its
    own gain plus the best of: a reachable candidate up to ``MAX_LINK_GAP`` frames earlier
    (minus ``GAP_COST`` per skipped frame), a restart from any earlier candidate (minus
    ``RESTART_COST``), or nothing. The best final candidate is traced back. Frames the path
    skips have no ball.
    """
    if balls.empty:
        return balls.reset_index(drop=True)
    b = balls.sort_values([s.FRAME, s.CONF], ascending=[True, False]).reset_index(drop=True)
    x1, y1, x2, y2 = (b[c].to_numpy(dtype=np.float64) for c in s.BOX)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = float(np.median(((x2 - x1) + (y2 - y1)) / 2))
    frames = b[s.FRAME].to_numpy()
    near = near_people(b, people).to_numpy()
    gain = b[s.CONF].to_numpy(dtype=np.float64) - MIN_CONF + NEAR_PLAYER_BONUS * near

    score = np.empty(len(b))
    prev = np.full(len(b), -1)
    starts = np.searchsorted(frames, frames, side="left")  # first candidate of each frame
    best_before, best_before_arg = -np.inf, -1  # best score over all earlier frames
    for j in range(len(b)):
        if j == starts[j]:  # first candidate in a new frame: fold the previous frames in
            for i in range(starts[j - 1] if j else 0, j):
                if score[i] > best_before:
                    best_before, best_before_arg = score[i], i
        best, arg = 0.0, -1
        if best_before - RESTART_COST > best:
            best, arg = best_before - RESTART_COST, best_before_arg
        lo = np.searchsorted(frames, frames[j] - MAX_LINK_GAP, side="left")
        for i in range(lo, starts[j]):
            gap = frames[j] - frames[i]
            if np.hypot(cx[j] - cx[i], cy[j] - cy[i]) <= MAX_SPEED * width * gap:
                cand = score[i] - GAP_COST * (gap - 1)
                if cand > best:
                    best, arg = cand, i
        score[j], prev[j] = gain[j] + best, arg

    path = []
    j = int(np.argmax(score))
    if score[j] <= 0:  # nothing worth calling a ball
        return b.iloc[:0].reset_index(drop=True)
    while j >= 0:
        path.append(j)
        j = int(prev[j])
    return b.loc[path[::-1]].reset_index(drop=True)


def interpolate_ball(balls: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """Fill gaps of at most ``max_gap`` missing frames between ball detections.

    Expects at most one row per frame (see ``track_ball``). A gap is only filled if a real
    ball could have travelled it (``MAX_SPEED``), so a jump to an unrelated detection
    isn't drawn as a straight line. Interpolated rows get ``interpolated=True`` and NaN
    confidence; other gaps are left empty.
    """
    if balls.empty:
        return balls.assign(**{s.INTERPOLATED: pd.Series(dtype=bool)})

    by_frame = balls.set_index(s.FRAME)
    width = float(
        np.median(
            ((by_frame[s.BOX[2]] - by_frame[s.BOX[0]]) + (by_frame[s.BOX[3]] - by_frame[s.BOX[1]]))
            / 2
        )
    )
    full = by_frame.reindex(range(int(by_frame.index.min()), int(by_frame.index.max()) + 1))
    missing = full[s.BOX[0]].isna()

    # Consecutive missing frames share the cumsum value of the detection before them.
    run_id = (~missing).cumsum()
    gap_len = missing.groupby(run_id).transform("sum")
    cx = (full[s.BOX[0]] + full[s.BOX[2]]) / 2
    cy = (full[s.BOX[1]] + full[s.BOX[3]]) / 2
    # Distance from the detection before each gap to the one after it.
    jump = np.hypot(cx.bfill() - cx.ffill(), cy.bfill() - cy.ffill())
    reachable = jump <= MAX_SPEED * width * (gap_len + 1)
    fillable = missing & (gap_len <= max_gap) & reachable

    full[s.BOX] = full[s.BOX].interpolate(method="linear")
    full = full[~missing | fillable].copy()
    full[s.INTERPOLATED] = missing[~missing | fillable]
    full[s.LABEL] = s.BALL
    full[s.TRACK_ID] = -1
    return full.rename_axis(s.FRAME).reset_index()
