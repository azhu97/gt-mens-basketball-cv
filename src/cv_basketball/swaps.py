"""Undo tracker ID swaps where two people's boxes overlap, using their jersey colors.

When two players collide, their boxes merge, and when they separate the tracker can hand
each one the other's ID. Each track then follows one person before the collision and the
other after it. Its jersey votes change partway, but often only a few clean votes show
it, because the torso stays covered for much of the contact. ``segments.split_on_change``
needs a long run of them to cut a track, and it cuts each track on its own evidence.

This works on the pair instead. For every collision it compares the colors of both
tracks before and after it, and swaps the two tails when that makes both people's colors
more consistent. Two weak pieces of evidence (A went white to purple, B went purple to
white) add up to a clear swap.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s

UNKNOWN = -1


def pairwise_iou(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(N, N)`` IoU between ``(N, 4)`` xyxy boxes, 0 on the diagonal."""
    iw = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(
        boxes[:, None, 0], boxes[None, :, 0]
    )
    ih = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(
        boxes[:, None, 1], boxes[None, :, 1]
    )
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
    np.fill_diagonal(iou, 0)
    return np.asarray(iou, dtype=np.float64)


def max_iou(people: pd.DataFrame) -> "pd.Series[float]":
    """Largest IoU of each row's box with another box in the same frame (0 if alone)."""
    out = pd.Series(0.0, index=people.index)
    for _, frame in people.groupby(s.FRAME):
        if len(frame) > 1:
            out.loc[frame.index] = pairwise_iou(frame[s.BOX].to_numpy(dtype=np.float64)).max(1)
    return out


@dataclass(frozen=True)
class Collision:
    """Two people's boxes overlapping over ``[start, end]``; ``rows`` are theirs at ``cut``."""

    start: int
    end: int
    cut: int  # frame of greatest overlap: the tails after it are what may be swapped
    rows: tuple[int, int]  # index labels of the two rows at ``cut``


def find_collisions(rows: pd.DataFrame, min_iou: float, max_gap: int = 15) -> list[Collision]:
    """Spans where two tracks' boxes overlap by over ``min_iou``, sorted by cut frame.

    Overlaps of the same pair up to ``max_gap`` frames apart count as one collision: the
    overlap of two players in contact dips and returns, and the swap is at its peak.
    """
    hits: list[tuple[int, int, int, float, int, int]] = []
    tracked = rows[rows[s.TRACK_ID] >= 0]
    for _, frame in tracked.groupby(s.FRAME):
        if len(frame) < 2:
            continue
        iou = pairwise_iou(frame[s.BOX].to_numpy(dtype=np.float64))
        ids = frame[s.TRACK_ID].to_numpy(dtype=np.int64)
        f = int(frame[s.FRAME].iloc[0])
        for i, j in zip(*np.nonzero(np.triu(iou) > min_iou), strict=True):
            lo, hi = (i, j) if ids[i] < ids[j] else (j, i)
            row_lo, row_hi = int(frame.index[lo]), int(frame.index[hi])
            hits.append((int(ids[lo]), int(ids[hi]), f, float(iou[i, j]), row_lo, row_hi))

    collisions = []
    by_pair = pd.DataFrame(hits, columns=["a", "b", "frame", "iou", "ra", "rb"])
    for _, pair in by_pair.groupby(["a", "b"]):
        pair = pair.sort_values("frame")
        span = (pair["frame"].diff() > max_gap).cumsum()
        for _, run in pair.groupby(span):
            peak = run.iloc[int(np.argmax(run["iou"].to_numpy()))]
            collisions.append(
                Collision(
                    int(run["frame"].min()),
                    int(run["frame"].max()),
                    int(peak["frame"]),
                    (int(peak["ra"]), int(peak["rb"])),
                )
            )
    return sorted(collisions, key=lambda c: c.cut)


def _consistency(counts: NDArray[np.float64], vote_accuracy: float) -> float:
    """Log-likelihood of one person's vote counts, under their best-fitting value.

    Each vote is taken to be right with probability ``vote_accuracy`` (and wrong
    otherwise), so a person whose votes all agree scores highest.
    """
    n = counts.sum()
    if n == 0:
        return 0.0
    right, wrong = np.log(vote_accuracy), np.log(1 - vote_accuracy)
    return float(counts.max() * right + (n - counts.max()) * wrong)


def repair_swaps(
    rows: pd.DataFrame,
    votes: "pd.Series[int]",
    n_values: int,
    min_iou: float = 0.15,
    window: int = 60,
    margin: float = 30.0,
    vote_accuracy: float = 0.9,
    max_gap: int = 15,
) -> "pd.Series[int]":
    """Return ``rows``' track IDs with each colliding pair's tails swapped where needed.

    ``votes`` holds each row's category (jersey cluster) in ``[0, n_values)`` or
    ``UNKNOWN``. For each collision, the votes within ``window`` frames before it and
    after it are pooled per person, once as the tracker had them and once swapped.
    The tails are swapped when that raises the log-likelihood (see ``_consistency``) by
    more than ``margin``. Overlaps up to ``max_gap`` frames apart are one collision.
    Collisions are handled in time order, each on the IDs as the earlier ones left them.
    """
    track_ids = rows[s.TRACK_ID].copy()
    frames = rows[s.FRAME].to_numpy()
    vote_arr = np.asarray(votes.loc[rows.index], dtype=np.int64)

    def counts(track: int, lo: int, hi: int) -> NDArray[np.float64]:
        mask = (track_ids.to_numpy() == track) & (frames >= lo) & (frames <= hi)
        v = vote_arr[mask]
        return np.bincount(v[v != UNKNOWN], minlength=n_values).astype(np.float64)

    for c in find_collisions(rows, min_iou, max_gap):
        a, b = int(track_ids.loc[c.rows[0]]), int(track_ids.loc[c.rows[1]])
        if a == b:
            continue
        a_before, b_before = (
            counts(a, c.start - window, c.start - 1),
            counts(b, c.start - window, c.start - 1),
        )
        a_after, b_after = (
            counts(a, c.end + 1, c.end + window),
            counts(b, c.end + 1, c.end + window),
        )
        keep = _consistency(a_before + a_after, vote_accuracy) + _consistency(
            b_before + b_after, vote_accuracy
        )
        swap = _consistency(a_before + b_after, vote_accuracy) + _consistency(
            b_before + a_after, vote_accuracy
        )
        if swap - keep > margin:
            tail = frames > c.cut
            a_tail = tail & (track_ids.to_numpy() == a)
            b_tail = tail & (track_ids.to_numpy() == b)
            track_ids.iloc[np.flatnonzero(a_tail)] = b
            track_ids.iloc[np.flatnonzero(b_tail)] = a
    return track_ids
