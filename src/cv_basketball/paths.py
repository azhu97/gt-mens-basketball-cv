"""Smooth each player's path on the court.

A player's court position is the bottom-centre of their box, projected to the floor. The
box's bottom edge jumps a few pixels from frame to frame (a foot lifts, a leg is cut
off, the detector's box wobbles), and far from the camera a few pixels are tens of
centimetres. Raw paths therefore shake: the median player "accelerates" at over
30 m/s², several times what a sprinting player can.

Each run of consecutive frames of a track is filtered on its own: a short median filter
removes single-frame spikes, then a Savitzky-Golay filter (a local quadratic fit over a
sliding window) removes the shake. Unlike a moving average, it follows a player who
changes direction without cutting the corner or lagging behind.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from cv_basketball import schema as s
from cv_basketball.court import CourtSpec
from cv_basketball.swaps import pairwise_iou


def _smooth(xy: np.ndarray, median: int, window: int) -> np.ndarray:
    """Median then Savitzky-Golay filter an ``(N, 2)`` path, shrinking windows to fit."""
    n = len(xy)
    if n >= 3 and median > 1:
        xy = median_filter(xy, size=(min(median, n - (n + 1) % 2), 1), mode="nearest")
    w = min(window, n if n % 2 else n - 1)
    if w >= 5:
        xy = savgol_filter(xy, w, polyorder=2, axis=0, mode="interp")
    return xy


def smooth_paths(tracks: pd.DataFrame, median: int = 5, window: int = 15) -> pd.DataFrame:
    """Return ``tracks`` with players' court coordinates smoothed along each track.

    Windows are in frames (odd). Only player and referee rows with court coordinates
    change; a gap in a track starts a new run, so a player isn't smoothed across it.
    """
    out = tracks.copy()
    people = out[(out[s.LABEL] != s.BALL) & out[s.COURT[0]].notna() & (out[s.TRACK_ID] >= 0)]
    for _, track in people.sort_values(s.FRAME).groupby(s.TRACK_ID):
        runs = (track[s.FRAME].diff() != 1).cumsum()
        for _, run in track.groupby(runs):
            xy = run[s.COURT].to_numpy(dtype=np.float64)
            out.loc[run.index, s.COURT] = _smooth(xy, median, window)
    return out


def fill_gaps(people: pd.DataFrame, max_gap: int = 30, max_iou: float = 0.3) -> pd.DataFrame:
    """Return ``people`` with short gaps inside each track filled by interpolated rows.

    The tracker keeps a player's ID through a brief occlusion, but has no box for the
    frames in between, so the dot blinks off. Gaps of up to ``max_gap`` frames get a
    linearly interpolated box, the track's label and team, and ``INTERPOLATED=True``;
    their color and vote columns are empty, so they never vote. A filled box that another
    box in its frame overlaps by more than ``max_iou`` is left out: that person is
    already on the court under another ID, and a second dot would double them.
    """
    fills = []
    for tid, track in people[people[s.TRACK_ID] >= 0].groupby(s.TRACK_ID):
        track = track.sort_values(s.FRAME)
        frames = track[s.FRAME].to_numpy()
        gap = np.diff(frames) - 1
        if not ((gap > 0) & (gap <= max_gap)).any():
            continue
        missing = np.concatenate(
            [
                np.arange(a + 1, b)
                for a, b, g in zip(frames, frames[1:], gap, strict=False)
                if 0 < g <= max_gap
            ]
        )
        boxes = {c: np.interp(missing, frames, track[c].to_numpy(dtype=np.float64)) for c in s.BOX}
        # carry the nearest earlier row's identity columns into the gap
        before = track.iloc[np.searchsorted(frames, missing) - 1]
        fill = pd.DataFrame(
            {
                s.FRAME: missing,
                s.TRACK_ID: tid,
                s.LABEL: before[s.LABEL].to_numpy(),
                **boxes,
            }
        )
        for col in (s.TEAM, s.TRACKER_ID):
            if col in track:
                fill[col] = before[col].to_numpy()
        fills.append(fill)
    if not fills:
        return people.assign(**{s.INTERPOLATED: False})
    filled = pd.concat(fills, ignore_index=True).assign(**{s.INTERPOLATED: True})
    by_frame = dict(tuple(people.groupby(s.FRAME)))
    keep = np.ones(len(filled), dtype=bool)
    for i, row in enumerate(filled[[s.FRAME, *s.BOX]].to_numpy()):
        others = by_frame.get(int(row[0]))
        if others is not None:
            stacked = np.vstack([row[1:], others[s.BOX].to_numpy(dtype=np.float64)])
            keep[i] = pairwise_iou(stacked)[0, 1:].max() <= max_iou
    filled = filled[keep]
    if s.TEAM_VOTE in people:
        filled[s.TEAM_VOTE] = -1
    return pd.concat([people.assign(**{s.INTERPOLATED: False}), filled], ignore_index=True)


def _foot(row: pd.Series) -> np.ndarray:
    return np.array([(row["x1"] + row["x2"]) / 2, row["y2"]], dtype=np.float64)


def stitch_into_gaps(
    people: pd.DataFrame, max_gap: int = 90, reach: float = 0.75, lead: int = 30
) -> pd.DataFrame:
    """Return ``people`` with short tracks that cover another track's gap merged into it.

    While the tracker has lost a player it may start a new ID for them, often as a
    second box on them shortly before the old ID drops out, then go back to the old ID:
    one person, two tracks, and two dots. A track B that ends inside a gap (up to
    ``max_gap`` frames) of track A, starts at most ``lead`` frames before that gap, is
    within ``reach`` box heights of A where the gap starts and where it ends, and doesn't
    contradict A's team, becomes part of A. B's rows in frames where A has a box are
    dropped (they were the duplicate box).
    """
    out = people.copy()
    tracked = out[out[s.TRACK_ID] >= 0].sort_values(s.FRAME)
    spans = tracked.groupby(s.TRACK_ID)[s.FRAME].agg(["min", "max"])
    team = tracked.groupby(s.TRACK_ID)[s.TEAM].first() if s.TEAM in tracked else None
    used: set[int] = set()
    drop = pd.Series(False, index=out.index)
    for a, track in tracked.groupby(s.TRACK_ID):
        frames = track[s.FRAME].to_numpy()
        steps = np.diff(frames)
        for i in np.flatnonzero((steps > 1) & (steps <= max_gap + 1)):
            before, after = track.iloc[i], track.iloc[i + 1]
            gap_start, gap_end = int(before[s.FRAME]), int(after[s.FRAME])
            height = before["y2"] - before["y1"]
            cands = spans[
                (spans["max"] > gap_start)
                & (spans["max"] < gap_end)
                & (spans["min"] >= gap_start - lead)
                & (spans["min"] > frames[0])
            ]
            best, best_cost = None, np.inf
            for b in cands.index:
                if b in used or b == a:
                    continue
                if team is not None and team[b] not in (-1, team[a]):
                    continue
                rows_b = tracked[tracked[s.TRACK_ID] == b]
                in_gap = rows_b[rows_b[s.FRAME] > gap_start]
                if in_gap.empty:
                    continue
                cost = np.linalg.norm(_foot(in_gap.iloc[0]) - _foot(before)) + np.linalg.norm(
                    _foot(rows_b.iloc[-1]) - _foot(after)
                )
                if cost <= 2 * reach * height and cost < best_cost:
                    best, best_cost = b, cost
            if best is not None:
                used.add(int(best))
                rows = out[s.TRACK_ID] == best
                drop |= rows & out[s.FRAME].isin(frames)
                out.loc[rows, s.TRACK_ID] = a
                if team is not None:
                    out.loc[rows, s.TEAM] = team[a]
    return out[~drop]


def off_court_people(
    tracks: pd.DataFrame, spec: CourtSpec, margin: float = 0.5, max_share: float = 0.5
) -> "pd.Series[bool]":
    """Rows of teamless player tracks that are mostly off the court: bench, coaches, staff.

    A track with no team is not in a team jersey, and if more than ``max_share`` of its
    positions are over ``margin`` metres outside the lines, it isn't playing. Tracks
    with a team stay (an inbounding player, or the court mapping drifting).
    """
    rows = (tracks[s.LABEL] == s.PLAYER) & (tracks[s.TEAM] == -1) & tracks[s.COURT[0]].notna()
    x, y = tracks[s.COURT[0]], tracks[s.COURT[1]]
    outside = (x < -margin) | (x > spec.length + margin) | (y < -margin) | (y > spec.width + margin)
    share = outside[rows].astype(float).groupby(tracks.loc[rows, s.TRACK_ID]).mean()
    off_tracks = share.index[share.to_numpy() > max_share]
    return rows & tracks[s.TRACK_ID].isin(off_tracks)


def foot_y(people: pd.DataFrame, window: int = 31, min_ratio: float = 0.85) -> "pd.Series[float]":
    """Image y of each person's feet, restored where the box is cut off at the bottom.

    When another player hides someone's legs, the detector's box stops at the knees and
    the projected feet jump metres towards the camera's far side. Per track, a box
    shorter than ``min_ratio`` of the track's rolling median height (over ``window``
    frames) whose bottom edge moved more than its top edge is cut at the bottom; its
    feet are put at its top plus the median height instead.
    """
    feet = people["y2"].astype(float).copy()
    for _, track in people[people[s.TRACK_ID] >= 0].sort_values(s.FRAME).groupby(s.TRACK_ID):
        if len(track) < 5:
            continue
        y1, y2 = track["y1"].astype(float), track["y2"].astype(float)

        def rolling_median(v: "pd.Series[float]") -> "pd.Series[float]":
            return v.rolling(window, center=True, min_periods=5).median()

        height = rolling_median(y2 - y1)
        top_shift = (y1 - rolling_median(y1)).abs()
        bottom_shift = (y2 - rolling_median(y2)).abs()
        cut = ((y2 - y1) < min_ratio * height) & (bottom_shift > top_shift)
        feet.loc[cut[cut].index] = (y1 + height)[cut]
    return feet
