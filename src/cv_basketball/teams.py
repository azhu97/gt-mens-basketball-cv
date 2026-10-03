"""Team assignment: cluster jersey colors, split tracks at team changes, vote per track."""

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from cv_basketball import schema as s
from cv_basketball.segments import split_on_change

UNKNOWN_TEAM = -1
_MAX_FIT_SAMPLES = 5000
# Upper-torso band as fractions of the box: central 50% width, 15-50% height. Avoids
# background, head and shorts.
_TORSO = np.array([0.25, 0.15, 0.75, 0.50])


def torso_boxes(xyxy: NDArray[np.float64]) -> NDArray[np.float64]:
    """Upper-torso region of each ``(N, 4)`` person box, in pixels."""
    x1, y1, x2, y2 = xyxy.T
    size = np.stack([x2 - x1, y2 - y1, x2 - x1, y2 - y1], axis=1)
    return np.asarray(np.stack([x1, y1, x1, y1], axis=1) + _TORSO * size, dtype=np.float64)


def torso_color_feature(frame: NDArray[np.uint8], xyxy: NDArray[np.float32]) -> NDArray[np.float64]:
    """Mean LAB color of the upper-torso region of a player box (NaNs if the crop is empty)."""
    tx1, ty1, tx2, ty2 = torso_boxes(np.asarray(xyxy, dtype=np.float64).reshape(1, 4))[0]
    cx1, cx2, cy1, cy2 = int(tx1), int(tx2), int(ty1), int(ty2)
    fh, fw = frame.shape[:2]
    cx1, cx2 = max(cx1, 0), min(cx2, fw)
    cy1, cy2 = max(cy1, 0), min(cy2, fh)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.full(3, np.nan)
    crop = frame[cy1:cy2, cx1:cx2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return np.asarray(lab.reshape(-1, 3).mean(axis=0), dtype=np.float64)


def occluded_torsos(people: pd.DataFrame, max_overlap: float = 0.3) -> "pd.Series[bool]":
    """Flag rows whose upper torso is more than ``max_overlap`` covered by another person's box.

    The color sample of such a row may be the other person's jersey. Which of the two is
    in front is unknown, so both are flagged when their boxes cover each other's torso.
    """
    occluded = pd.Series(False, index=people.index)
    for _, frame in people.groupby(s.FRAME):
        if len(frame) < 2:
            continue
        boxes = frame[s.BOX].to_numpy(dtype=np.float64)
        torsos = torso_boxes(boxes)
        # (torso i, box j) intersection widths and heights
        iw = np.minimum(torsos[:, None, 2], boxes[None, :, 2]) - np.maximum(
            torsos[:, None, 0], boxes[None, :, 0]
        )
        ih = np.minimum(torsos[:, None, 3], boxes[None, :, 3]) - np.maximum(
            torsos[:, None, 1], boxes[None, :, 1]
        )
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        np.fill_diagonal(inter, 0)
        area = (torsos[:, 2] - torsos[:, 0]) * (torsos[:, 3] - torsos[:, 1])
        covered = inter.max(axis=1) / np.maximum(area, 1e-9)
        occluded.loc[frame.index] = covered > max_overlap
    return occluded


def cluster_jerseys(
    players: pd.DataFrame,
    occluded: "pd.Series[bool] | None" = None,
    n_teams: int = 2,
    seed: int = 0,
) -> "pd.Series[int]":
    """Return a jersey-color cluster per row of ``players`` (``UNKNOWN_TEAM`` if unclusterable).

    Rows without a track ID or a color feature, or flagged in ``occluded``, are not
    clustered.
    """
    clusters = pd.Series(UNKNOWN_TEAM, index=players.index, dtype="int64")
    feats = players[s.COLOR_FEATURES].to_numpy(dtype=np.float64)
    valid = (players[s.TRACK_ID].to_numpy() >= 0) & ~np.isnan(feats).any(axis=1)
    if occluded is not None:
        valid &= ~occluded.loc[players.index].to_numpy(dtype=bool)
    valid_feats = feats[valid]
    if len(np.unique(valid_feats, axis=0)) < n_teams:
        return clusters

    rng = np.random.default_rng(seed)
    fit_feats = (
        valid_feats[rng.choice(len(valid_feats), _MAX_FIT_SAMPLES, replace=False)]
        if len(valid_feats) > _MAX_FIT_SAMPLES
        else valid_feats
    )
    km = KMeans(n_clusters=n_teams, n_init=10, random_state=seed).fit(fit_feats)
    clusters.loc[valid] = km.predict(valid_feats)
    return clusters


def vote_teams(track_ids: "pd.Series[int]", clusters: "pd.Series[int]") -> "pd.Series[int]":
    """Majority-vote ``clusters`` within each track; rows without a track ID stay unknown."""
    team = pd.Series(UNKNOWN_TEAM, index=track_ids.index, dtype="int64")
    known = clusters != UNKNOWN_TEAM
    per_track = clusters[known].groupby(track_ids[known]).agg(lambda v: int(v.mode().iloc[0]))
    tracked = track_ids >= 0
    team.loc[tracked] = track_ids[tracked].map(per_track).fillna(UNKNOWN_TEAM).astype("int64")
    return team


def assign_teams(
    players: pd.DataFrame,
    next_track_id: int,
    occluded: "pd.Series[bool] | None" = None,
    n_teams: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``track_id`` (split at team changes) and ``team`` for each row of ``players``.

    Team is constant within each returned track ID. Rows flagged in ``occluded`` don't
    vote. Referees and bench players are not handled and will be forced into a team.
    """
    clusters = cluster_jerseys(players, occluded, n_teams, seed)
    track_ids = split_on_change(players, clusters, next_track_id, n_teams)
    return pd.DataFrame({s.TRACK_ID: track_ids, s.TEAM: vote_teams(track_ids, clusters)})
