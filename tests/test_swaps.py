import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.diagnostics import flip_report
from cv_basketball.swaps import find_collisions, pairwise_iou, repair_swaps

RED, BLUE, NONE = 0, 1, -1


def _crossing(swap: bool, hidden: int = 0) -> tuple[pd.DataFrame, "pd.Series[int]"]:
    """Two people walk through each other at frame 50, with votes along their own path.

    Person RED walks right, person BLUE walks left; their boxes meet at frame 50. With
    ``swap``, tracker ID 1 follows RED until the collision and BLUE after it (and ID 2
    the reverse). The first ``hidden`` frames after the collision have no clean vote.
    """
    rows, votes = [], []
    for f in range(100):
        red_x, blue_x = 2.0 * f, 200.0 - 2.0 * f
        after = f > 50
        for person, x in ((RED, red_x), (BLUE, blue_x)):
            tid = (
                (1 if person == RED else 2) if not (swap and after) else (2 if person == RED else 1)
            )
            rows.append(
                {s.FRAME: f, s.TRACK_ID: tid, "x1": x, "y1": 0.0, "x2": x + 40, "y2": 100.0}
            )
            visible = not (40 <= f <= 60) and not (after and f <= 60 + hidden)
            votes.append(person if visible else NONE)
    df = pd.DataFrame(rows)
    return df, pd.Series(votes, index=df.index)


def test_pairwise_iou() -> None:
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 110, 110]], dtype=float)
    iou = pairwise_iou(boxes)
    assert np.isclose(iou[0, 1], 1 / 3)
    assert iou[0, 2] == 0 and iou[0, 0] == 0


def test_collision_is_found_once_at_peak_overlap() -> None:
    rows, _ = _crossing(swap=False)
    (c,) = find_collisions(rows, min_iou=0.15)
    assert c.start < 50 < c.end
    assert c.cut == 50


def test_swapped_tails_are_swapped_back() -> None:
    rows, votes = _crossing(swap=True)
    ids = repair_swaps(rows, votes, n_values=2, margin=5.0)
    # each repaired ID now votes for one person only
    per_id = votes[votes != NONE].groupby(ids[votes != NONE]).nunique()
    assert (per_id == 1).all()


def test_tracks_that_did_not_swap_are_left_alone() -> None:
    rows, votes = _crossing(swap=False)
    ids = repair_swaps(rows, votes, n_values=2, margin=5.0)
    assert ids.equals(rows[s.TRACK_ID])


def test_one_sided_evidence_below_margin_is_not_swapped() -> None:
    # After the swap only 3 clean frames remain within the window: too little to act on.
    rows, votes = _crossing(swap=True, hidden=36)
    ids = repair_swaps(rows, votes, n_values=2, window=40, margin=30.0)
    assert ids.equals(rows[s.TRACK_ID])


def test_flip_report_counts_visible_flip_at_a_split() -> None:
    # Track 1 is red until frame 9; track 2 starts on the same box at frame 10, blue.
    rows = [
        {
            s.FRAME: f,
            s.TRACK_ID: 1 if f < 10 else 2,
            s.TRACKER_ID: 1,
            s.LABEL: s.PLAYER,
            s.TEAM: 0 if f < 10 else 1,
            s.TEAM_VOTE: 0 if f < 10 else 1,
            s.MAX_IOU: 0.0,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 20.0,
        }
        for f in range(20)
    ]
    report = flip_report(pd.DataFrame(rows))
    assert report.visible_flips == 1
    assert report.tracker_flips == 1
    assert report.tracker_flips_vote_unchanged == 0
    assert report.wrong_votes == 0
