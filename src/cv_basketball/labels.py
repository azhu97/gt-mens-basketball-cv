"""Person label smoothing: one label (player or referee) per person, by majority vote."""

import pandas as pd

from cv_basketball import schema as s
from cv_basketball.segments import split_on_change
from cv_basketball.tracking import NO_TRACK


def vote_person_labels(
    people: pd.DataFrame, next_track_id: int, split_window: int = 15, split_min_run: int = 30
) -> pd.DataFrame:
    """Return ``track_id`` (split at label changes) and ``label`` for each row of ``people``.

    The detector classifies each frame independently, so a borderline person flips
    between player and referee. Tracks are first cut where the label changes for good
    (an ID swap between a player and a referee), then each track takes its most frequent
    label (ties go to player). Rows without a track ID keep their per-frame label.
    """
    is_ref = (people[s.LABEL] == s.REFEREE).astype("int64")
    track_ids = split_on_change(people, is_ref, next_track_id, 2, split_window, split_min_run)
    labels = people[s.LABEL].copy()
    tracked = track_ids != NO_TRACK
    ref_tracks = is_ref[tracked].astype(float).groupby(track_ids[tracked]).mean() > 0.5
    labels.loc[tracked] = track_ids[tracked].map(ref_tracks).map({True: s.REFEREE, False: s.PLAYER})
    return pd.DataFrame({s.TRACK_ID: track_ids, s.LABEL: labels})
