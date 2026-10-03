"""Team-label stability report for a tracks table (``cvb flips``).

Every output track has one team, so team flicker shows up in two other ways:

- a *visible flip*: a track ends and another starts in the same spot with the other
  team, so the dot on the court changes color (this is what splits look like);
- a *tracker flip*: the team shown along the tracker's own ID changes. After an ID swap
  that's correct (we follow the person, not the ID). When the row's own jersey vote didn't
  change across it, the flip is a classification error instead.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from cv_basketball import schema as s
from cv_basketball.swaps import pairwise_iou

TEAMS = (0, 1)


@dataclass
class FlipReport:
    tracks: int  # player tracks in the output
    dot_flips: int  # a box changes color (incl. to/from no team) from one frame to the next
    visible_flips: int  # a track hands over to one of the other team in the same spot
    tracker_flips: int  # team changes along the tracker's own IDs
    tracker_flips_in_overlap: int  # ... with the box overlapping another nearby
    tracker_flips_vote_unchanged: int  # ... where the jersey votes agree on both sides
    wrong_votes: float  # share of clean jersey votes that disagree with the team shown
    frames_over_5: int  # frames where a team has more than 5 players
    frames_5v5: int  # frames with exactly 5 players on each team
    frames: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _majority(votes: "pd.Series[int]") -> int:
    known = votes[votes.isin(TEAMS)]
    return int(known.mode().iloc[0]) if len(known) else -1


def dot_flips(players: pd.DataFrame, min_iou: float = 0.3) -> int:
    """Times a player's box changes team color between consecutive frames.

    Boxes are matched frame to frame by IoU (best one-to-one assignment), whatever their
    track IDs, so this counts what a viewer sees: a dot changing color, including going
    to or from grey (no team).
    """
    flips = 0
    prev: pd.DataFrame | None = None
    for _, cur in players.groupby(s.FRAME):
        if prev is not None and int(prev[s.FRAME].iloc[0]) == int(cur[s.FRAME].iloc[0]) - 1:
            boxes = np.vstack([prev[s.BOX], cur[s.BOX]]).astype(np.float64)
            iou = pairwise_iou(boxes)[: len(prev), len(prev) :]
            rows, cols = linear_sum_assignment(-iou)
            matched = iou[rows, cols] >= min_iou
            before = prev[s.TEAM].to_numpy()[rows[matched]]
            after = cur[s.TEAM].to_numpy()[cols[matched]]
            flips += int((before != after).sum())
        prev = cur
    return flips


def flip_report(
    tracks: pd.DataFrame, overlap_iou: float = 0.15, near_frames: int = 15, handover: int = 3
) -> FlipReport:
    """Measure team-label stability of ``tracks`` (needs the diagnostic columns).

    A flip is "in overlap" if the box overlaps another by more than ``overlap_iou``
    within ``near_frames`` of it. A visible flip is a track ending and one of the
    other team starting within ``handover`` frames whose first box overlaps its last
    box by IoU 0.3 or more.
    """
    players = tracks[(tracks[s.LABEL] == s.PLAYER) & (tracks[s.TRACK_ID] >= 0)]
    on_team = players[players[s.TEAM].isin(TEAMS)].sort_values(s.FRAME)

    tracker_flips = in_overlap = vote_unchanged = 0
    for _, track in on_team.groupby(s.TRACKER_ID):
        team = track[s.TEAM].to_numpy()
        frames = track[s.FRAME].to_numpy()
        for i in np.flatnonzero(np.diff(team)) + 1:
            tracker_flips += 1
            near = track[np.abs(frames - frames[i]) <= near_frames]
            in_overlap += bool((near[s.MAX_IOU] > overlap_iou).any())
            before = _majority(track[s.TEAM_VOTE].iloc[max(i - 30, 0) : i])
            after = _majority(track[s.TEAM_VOTE].iloc[i : i + 30])
            vote_unchanged += before == after != -1

    visible = 0
    spans = on_team.groupby(s.TRACK_ID)[s.FRAME].agg(["min", "max"])
    first = on_team.drop_duplicates(s.TRACK_ID, keep="first").set_index(s.TRACK_ID)
    last = on_team.drop_duplicates(s.TRACK_ID, keep="last").set_index(s.TRACK_ID)
    for tid, end in spans["max"].items():
        starts = spans.index[(spans["min"] > end) & (spans["min"] <= end + handover)]
        others = [t for t in starts if first.at[t, s.TEAM] != last.at[tid, s.TEAM]]
        if not others:
            continue
        boxes = np.vstack([last.loc[[tid], s.BOX], first.loc[others, s.BOX]]).astype(float)
        visible += bool((pairwise_iou(boxes)[0, 1:] >= 0.3).any())

    clean = on_team[on_team[s.TEAM_VOTE].isin(TEAMS)]
    wrong = float((clean[s.TEAM_VOTE] != clean[s.TEAM]).mean()) if len(clean) else 0.0

    n_frames = int(tracks[s.FRAME].max()) + 1 if len(tracks) else 0
    sizes = on_team.groupby([s.FRAME, s.TEAM]).size().unstack(fill_value=0)
    sizes = sizes.reindex(index=range(n_frames), columns=list(TEAMS), fill_value=0)
    return FlipReport(
        tracks=int(players[s.TRACK_ID].nunique()),
        dot_flips=dot_flips(players),
        visible_flips=visible,
        tracker_flips=tracker_flips,
        tracker_flips_in_overlap=in_overlap,
        tracker_flips_vote_unchanged=vote_unchanged,
        wrong_votes=round(wrong, 4),
        frames_over_5=int((sizes > 5).any(axis=1).sum()),
        frames_5v5=int((sizes == 5).all(axis=1).sum()),
        frames=n_frames,
    )
