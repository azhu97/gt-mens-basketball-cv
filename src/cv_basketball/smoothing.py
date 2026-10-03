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


def grey_out_sixth_players(
    players: pd.DataFrame, max_per_team: int = 5, window: int = 30
) -> "pd.Series[int]":
    """Team per row with the weakest extra players in over-full frames set to unknown.

    A team can't have more than ``max_per_team`` players on the court, so in a frame
    with more, someone's team is wrong. Crowded frames of one team are grouped into
    stretches (gaps up to ``window`` frames); within a stretch, tracks are ranked by
    their support, meaning clean votes for the team minus votes against it, from
    ``window`` frames before the stretch to ``window`` after. In each crowded frame,
    the lowest-ranked tracks (as many as the stretch's worst excess) lose their team for
    the whole stretch, so the same player is changed throughout. If the other team has
    room for them in every frame of the stretch, they join it (typically a player
    hidden behind an opponent who took the opponent's colour); otherwise they turn grey,
    which is better than the wrong colour.
    """
    team = players[s.TEAM].copy()
    on_team = players[team.isin([0, 1])]
    sizes = on_team.groupby([s.FRAME, s.TEAM])[s.TRACK_ID].transform("size")
    crowded = on_team[sizes > max_per_team]
    votes = players[s.TEAM_VOTE]
    for _, rows in crowded.groupby(s.TEAM):
        tm = int(rows[s.TEAM].iloc[0])
        frames = np.sort(rows[s.FRAME].unique())
        stretch = np.r_[0, np.cumsum(np.diff(frames) > window)]
        for k in np.unique(stretch):
            lo, hi = frames[stretch == k][[0, -1]]
            near = players[
                players[s.FRAME].between(lo - window, hi + window) & (players[s.TEAM] == tm)
            ]
            v = votes.loc[near.index]
            score = (v == tm).astype(int) - ((v != tm) & v.isin([0, 1])).astype(int)
            support = score.groupby(near[s.TRACK_ID]).sum()
            # the weakest tracks among the crowded frames, grey for the whole stretch
            in_stretch = rows[rows[s.FRAME].between(lo, hi)]
            extra = int(in_stretch.groupby(s.FRAME).size().max()) - max_per_team
            present = support.loc[in_stretch[s.TRACK_ID].unique()].sort_values(kind="stable")
            weakest = present.index[:extra]
            span = players[s.FRAME].between(lo, hi) & players[s.TRACK_ID].isin(weakest)
            # if the other team is short throughout, that's where the extras belong
            other = 1 - tm
            stretch_frames = players[players[s.FRAME].between(lo, hi)]
            other_size = (stretch_frames[s.TEAM] == other).groupby(stretch_frames[s.FRAME]).sum()
            fits = int(other_size.max()) + extra <= max_per_team
            team[span & (team == tm)] = other if fits else UNKNOWN_TEAM
    return team


def fill_to_five(
    players: pd.DataFrame,
    max_per_team: int = 5,
    min_full_share: float = 0.5,
    min_room_share: float = 0.8,
) -> "pd.Series[int]":
    """Team per row, with grey players put on the team that is short of five.

    A player whose jersey is hidden most of the time (standing behind someone) gets too
    few clean votes for a team and shows grey. But the court has five of each, so a
    grey track joins a team if both hold:

    - joining never makes that team more than ``max_per_team`` in any of its frames,
      and the team never shows more players than the other one there;
    - in at least ``min_full_share`` of its frames, the other team already shows
      ``max_per_team`` (so the player can't be theirs).

    Tracks dressed like a referee (more referee-like votes than team votes) stay grey.
    Longest tracks are placed first, and counts are updated after each.
    """
    team = players[s.TEAM].copy()
    votes = players[s.TEAM_VOTE]
    grey = players[(team == UNKNOWN_TEAM) & (players[s.TRACK_ID] >= 0)]
    order = grey.groupby(s.TRACK_ID).size().sort_values(ascending=False).index
    for tid in order:
        rows = players[s.TRACK_ID] == tid
        v = votes[rows]
        if int((v == 2).sum()) > int(v.isin([0, 1]).sum()):
            continue
        frames = players.loc[rows, s.FRAME].unique()
        others = players[players[s.FRAME].isin(frames) & ~rows]
        counts = pd.crosstab(others[s.FRAME], team[others.index]).reindex(
            index=frames, columns=[0, 1], fill_value=0
        )
        for short, full in ((0, 1), (1, 0)):
            room = ((counts[short] < max_per_team) & (counts[short] <= counts[full])).mean()
            room = room >= min_room_share
            if room and (counts[full] >= max_per_team).mean() >= min_full_share:
                team[rows & (team == UNKNOWN_TEAM)] = short
                break
    return team
