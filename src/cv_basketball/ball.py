"""Ball post-processing: one ball per frame, short gaps filled by linear interpolation."""

import pandas as pd

from cv_basketball import schema as s


def select_ball(balls: pd.DataFrame) -> pd.DataFrame:
    """Keep only the highest-confidence ball detection in each frame."""
    return (
        balls.sort_values(s.CONF, ascending=False)
        .drop_duplicates(s.FRAME)
        .sort_values(s.FRAME)
        .reset_index(drop=True)
    )


def interpolate_ball(balls: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """Fill gaps of at most ``max_gap`` missing frames between ball detections.

    Expects at most one row per frame (see ``select_ball``). Interpolated rows get
    ``interpolated=True`` and NaN confidence; longer gaps are left empty.
    """
    if balls.empty:
        return balls.assign(**{s.INTERPOLATED: pd.Series(dtype=bool)})

    by_frame = balls.set_index(s.FRAME)
    full = by_frame.reindex(range(int(by_frame.index.min()), int(by_frame.index.max()) + 1))
    missing = full[s.BOX[0]].isna()

    # Consecutive missing frames share the cumsum value of the detection before them.
    run_id = (~missing).cumsum()
    gap_len = missing.groupby(run_id).transform("sum")
    fillable = missing & (gap_len <= max_gap)

    full[s.BOX] = full[s.BOX].interpolate(method="linear")
    full = full[~missing | fillable].copy()
    full[s.INTERPOLATED] = missing[~missing | fillable]
    full[s.LABEL] = s.BALL
    full[s.TRACK_ID] = -1
    return full.rename_axis(s.FRAME).reset_index()
