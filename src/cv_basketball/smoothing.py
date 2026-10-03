"""Filter short-lived team noise along each tracker ID, after team assignment.

``segments.split_on_change`` cuts a tracker ID into pieces wherever its jersey votes
change for 30+ rows, and each piece votes its own team. Most cuts are real ID swaps,
but some are noise that lasts just long enough: motion blur on a fast break makes a
purple jersey look like a referee's for a second, or a covered torso samples the
neighbour's jersey. The dot then goes purple, grey, purple on the court.

Here the pieces of one tracker ID are read in order, like a sequence of labels, and a
piece is relabelled when both pieces beside it agree and it is too weak to stand:

- a grey (unknown) piece between two pieces of the same team takes that team, unless
  its own votes favour the other team; at the start or end of a tracker ID, a grey
  piece next to a team piece takes it only if its own votes favour that team;
- a piece of the other team takes its neighbours' team if it is short (a brief swap
  there and back is much rarer than a second of bad color samples).
"""

import numpy as np
import pandas as pd

from cv_basketball import schema as s

UNKNOWN_TEAM = -1


def _pieces(track: pd.DataFrame) -> list[pd.Index]:
    """Row index of each run of one tracker ID's rows (frame order) with the same track ID."""
    ids = track[s.TRACK_ID].to_numpy()
    starts = [0, *(np.flatnonzero(ids[1:] != ids[:-1]) + 1), len(ids)]
    return [track.index[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)]


def smooth_teams(players: pd.DataFrame, max_flip_rows: int = 45) -> "pd.Series[int]":
    """Return ``players``' team with noisy pieces relabelled (see module docstring).

    Needs ``TRACKER_ID``, ``TRACK_ID``, ``TEAM`` and ``TEAM_VOTE``. A piece of the other
    team is relabelled only if it has at most ``max_flip_rows`` rows. A relabelled
    piece's whole track takes the new team.
    """
    team = players[s.TEAM].copy()
    votes = players[s.TEAM_VOTE]
    for _, track in players[players[s.TRACKER_ID] >= 0].sort_values(s.FRAME).groupby(s.TRACKER_ID):
        pieces = _pieces(track)
        # a relabel can make a neighbour relabelable; a pass per piece is plenty (and
        # bounds the loop if two pieces of one track pull it different ways)
        for _ in range(len(pieces)):
            changed = False
            for i, piece in enumerate(pieces):
                sides = {
                    int(team.loc[pieces[j][0]]) for j in (i - 1, i + 1) if 0 <= j < len(pieces)
                }
                own = int(team.loc[piece[0]])
                if len(sides) != 1 or UNKNOWN_TEAM in sides or own in sides:
                    continue
                (around,) = sides
                edge = i == 0 or i == len(pieces) - 1
                piece_votes = votes.loc[piece]
                agree = int((piece_votes == around).sum())
                against = int(((piece_votes != around) & piece_votes.isin([0, 1])).sum())
                if own == UNKNOWN_TEAM:
                    relabel = agree > against if edge else agree >= against
                else:
                    relabel = not edge and len(piece) <= max_flip_rows
                if relabel:
                    # the whole track, so its team stays constant (with swap repair, a
                    # track can continue on another tracker ID's rows)
                    team[players[s.TRACK_ID] == players.at[piece[0], s.TRACK_ID]] = around
                    changed = True
            if not changed:
                break
    return team


def drop_blips(players: pd.DataFrame, min_rows: int = 10) -> "pd.Series[bool]":
    """Rows to keep: drops tracks with no team and under ``min_rows`` rows.

    These are mostly one-off boxes (a second box on a player, a hand, a bench player
    stepping in) that blink on the court for a few frames.
    """
    size = players.groupby(s.TRACK_ID)[s.FRAME].transform("size")
    return ~((players[s.TEAM] == UNKNOWN_TEAM) & (size < min_rows))
