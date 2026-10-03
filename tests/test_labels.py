import pandas as pd

from cv_basketball import schema as s
from cv_basketball.labels import vote_person_labels


def _people(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{s.FRAME: f, s.TRACK_ID: tid, s.LABEL: label} for f, (tid, label) in enumerate(rows)]
    )


def test_majority_label_wins_per_track() -> None:
    people = _people(
        [(8, s.REFEREE)] * 3 + [(8, s.PLAYER)] + [(5, s.PLAYER)] * 3 + [(5, s.REFEREE)]
    )
    labels = vote_person_labels(people, next_track_id=9)[s.LABEL]
    assert (labels[people[s.TRACK_ID] == 8] == s.REFEREE).all()
    assert (labels[people[s.TRACK_ID] == 5] == s.PLAYER).all()


def test_tie_goes_to_player() -> None:
    people = _people([(1, s.REFEREE), (1, s.PLAYER)])
    assert (vote_person_labels(people, next_track_id=2)[s.LABEL] == s.PLAYER).all()


def test_untracked_rows_keep_their_label() -> None:
    people = _people([(-1, s.REFEREE), (-1, s.PLAYER), (-1, s.PLAYER)])
    out = vote_person_labels(people, next_track_id=0)
    assert out[s.LABEL].tolist() == [s.REFEREE, s.PLAYER, s.PLAYER]
    assert (out[s.TRACK_ID] == -1).all()


def test_player_to_referee_swap_splits_track() -> None:
    # Track 8 follows a player for 60 frames, then a referee for 200 (an ID swap). A
    # whole-track vote would call the player a referee from frame 0.
    people = _people([(8, s.PLAYER)] * 60 + [(8, s.REFEREE)] * 200)
    out = vote_person_labels(people, next_track_id=20)
    assert out[s.TRACK_ID].tolist() == [8] * 60 + [20] * 200
    assert out[s.LABEL].tolist() == [s.PLAYER] * 60 + [s.REFEREE] * 200
