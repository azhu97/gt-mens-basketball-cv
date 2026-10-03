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
