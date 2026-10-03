"""Team assignment: cluster jersey colors, split tracks at team changes, vote per track."""

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

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
    """LAB histogram of the upper-torso region of a player box (NaNs if the crop is empty).

    See ``schema.COLOR_FEATURES`` for the layout and why it isn't a mean color.
    """
    tx1, ty1, tx2, ty2 = torso_boxes(np.asarray(xyxy, dtype=np.float64).reshape(1, 4))[0]
    fh, fw = frame.shape[:2]
    cx1, cx2 = max(int(tx1), 0), min(int(tx2), fw)
    cy1, cy2 = max(int(ty1), 0), min(int(ty2), fh)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.full(len(s.COLOR_FEATURES), np.nan)
    lab = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hist_l = np.histogram(lab[:, 0], s.HIST_BINS_L, (0, 256))[0]
    # a and b sit near 128 for jerseys under arena light; 96-176 keeps the bins useful
    hist_a = np.histogram(np.clip(lab[:, 1], 96, 175), s.HIST_BINS_AB, (96, 176))[0]
    hist_b = np.histogram(np.clip(lab[:, 2], 96, 175), s.HIST_BINS_AB, (96, 176))[0]
    return np.concatenate([hist_l, hist_a, hist_b]).astype(np.float64) / len(lab)


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


def _clean_features(
    rows: pd.DataFrame, occluded: "pd.Series[bool] | None"
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Color features of ``rows`` and which are usable (tracked, present, not occluded)."""
    feats = rows[s.COLOR_FEATURES].to_numpy(dtype=np.float64)
    valid = (rows[s.TRACK_ID].to_numpy() >= 0) & ~np.isnan(feats).any(axis=1)
    if occluded is not None:
        valid &= ~occluded.loc[rows.index].to_numpy(dtype=bool)
    return feats, valid


def referee_like(
    feats: NDArray[np.float64], player_feats: NDArray[np.float64], ref_feats: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether each row of ``feats`` is dressed more like the referees than the players.

    A logistic regression on the detector's own player/referee labels learns what sets
    the striped shirt apart. A distance to the mean referee color can't do that: purple
    jerseys in shadow are as close to it as the stripes are.
    """
    x = np.vstack([player_feats, ref_feats])
    y = np.r_[np.zeros(len(player_feats)), np.ones(len(ref_feats))]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)
    return np.asarray(lr.predict(feats) == 1, dtype=np.bool_)


def cluster_jerseys(
    players: pd.DataFrame,
    occluded: "pd.Series[bool] | None" = None,
    n_teams: int = 2,
    seed: int = 0,
    referees: pd.DataFrame | None = None,
) -> "pd.Series[int]":
    """Return a jersey-color cluster per row of ``players`` (``UNKNOWN_TEAM`` if unclusterable).

    Rows without a track ID or a color feature, or flagged in ``occluded``, are not
    clustered. With ``referees``, rows dressed like a referee (see ``referee_like``) get
    cluster ``n_teams``, which is not a team, and are left out of the team clustering.
    """
    clusters = pd.Series(UNKNOWN_TEAM, index=players.index, dtype="int64")
    feats, valid = _clean_features(players, occluded)
    if referees is not None:
        ref_feats, ref_valid = _clean_features(referees, occluded)
        if ref_valid.any() and valid.any():
            ref_like = valid.copy()
            ref_like[valid] = referee_like(feats[valid], feats[valid], ref_feats[ref_valid])
            clusters.loc[ref_like] = n_teams
            valid &= ~ref_like
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


def vote_teams(
    track_ids: "pd.Series[int]", clusters: "pd.Series[int]", n_teams: int = 2
) -> "pd.Series[int]":
    """Majority-vote ``clusters`` within each track; rows without a track ID stay unknown.

    A track that votes for a cluster that is not a team (``>= n_teams``) is unknown too.
    """
    team = pd.Series(UNKNOWN_TEAM, index=track_ids.index, dtype="int64")
    known = clusters != UNKNOWN_TEAM
    per_track = clusters[known].groupby(track_ids[known]).agg(lambda v: int(v.mode().iloc[0]))
    per_track = per_track.where(per_track < n_teams, UNKNOWN_TEAM)
    tracked = track_ids >= 0
    team.loc[tracked] = track_ids[tracked].map(per_track).fillna(UNKNOWN_TEAM).astype("int64")
    return team


def assign_teams(
    players: pd.DataFrame,
    next_track_id: int,
    occluded: "pd.Series[bool] | None" = None,
    n_teams: int = 2,
    seed: int = 0,
    referees: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``track_id`` (split at team changes) and ``team`` for each row of ``players``.

    Team is constant within each returned track ID. Rows flagged in ``occluded`` don't
    vote. With ``referees``, tracks (or the part of one) dressed like a referee, such as
    a referee the detector called a player, get ``UNKNOWN_TEAM`` instead of a team.
    Coaches and bench players often do too, since they aren't in a team jersey.
    """
    clusters = cluster_jerseys(players, occluded, n_teams, seed, referees)
    n_values = n_teams + (referees is not None)
    track_ids = split_on_change(players, clusters, next_track_id, n_values)
    team = vote_teams(track_ids, clusters, n_teams)
    return pd.DataFrame({s.TRACK_ID: track_ids, s.TEAM: team})
