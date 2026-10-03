"""Split tracks where a per-row category changes for good (a likely tracker ID swap).

When two people collide, the tracker can swap their IDs, so one track ends up following
two people. A per-row category that should be fixed for a person (jersey cluster,
player vs referee) then flips partway through the track. Cutting the track there lets a
per-track majority vote cover one person instead of two.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s

UNKNOWN = -1


def _stable_runs(values: NDArray[np.int64], n_values: int, window: int, min_run: int) -> list[int]:
    """Start indices of stable same-value runs in one track's frame-ordered values.

    Values are smoothed with a centred rolling majority over ``window`` rows, then runs
    shorter than ``min_run`` are absorbed into a neighbouring run, shortest first.
    """
    known = values != UNKNOWN
    if not known.any():
        return [0]
    one_hot = np.zeros((len(values), n_values))
    one_hot[known, values[known]] = 1
    votes = pd.DataFrame(one_hot).rolling(window, center=True, min_periods=1).sum().to_numpy()
    # A window with no known values carries the nearest smoothed value on, not value 0.
    majority = pd.Series(votes.argmax(axis=1)).where(votes.sum(axis=1) > 0)
    smoothed = majority.ffill().bfill().to_numpy(dtype=np.int64, copy=True)

    while True:
        starts = [0, *np.flatnonzero(np.diff(smoothed)) + 1]
        ends = [*starts[1:], len(smoothed)]
        lengths = np.subtract(ends, starts)
        short = [i for i, n in enumerate(lengths) if n < min_run]
        if len(starts) == 1 or not short:
            return starts
        i = min(short, key=lambda j: lengths[j])
        neighbour = starts[i - 1] if i > 0 else starts[i + 1]
        smoothed[starts[i] : ends[i]] = smoothed[neighbour]


def split_on_change(
    rows: pd.DataFrame,
    values: "pd.Series[int]",
    next_track_id: int,
    n_values: int = 2,
    window: int = 15,
    min_run: int = 30,
) -> "pd.Series[int]":
    """Return track IDs for ``rows``, cut wherever ``values`` changes for at least ``min_run`` rows.

    ``values`` holds ints in ``[0, n_values)`` or ``UNKNOWN``. Each run after a track's
    first gets a fresh ID from ``next_track_id`` upwards. Rows without a track ID are
    left alone.
    """
    track_ids = rows[s.TRACK_ID].copy()
    tracked = rows[rows[s.TRACK_ID] >= 0].sort_values(s.FRAME)
    for _, track in tracked.groupby(s.TRACK_ID):
        idx = track.index
        track_values = np.asarray(values.loc[idx], dtype=np.int64)
        starts = _stable_runs(track_values, n_values, window, min_run)
        ends = [*starts[1:], len(idx)]
        for start, end in zip(starts[1:], ends[1:], strict=True):
            track_ids.loc[idx[start:end]] = next_track_id
            next_track_id += 1
    return track_ids
