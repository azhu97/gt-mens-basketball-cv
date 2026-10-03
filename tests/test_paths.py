import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.paths import smooth_paths


def _track(xy: np.ndarray, frames: np.ndarray | None = None) -> pd.DataFrame:
    frames = np.arange(len(xy)) if frames is None else frames
    return pd.DataFrame(
        {
            s.FRAME: frames,
            s.TRACK_ID: 1,
            s.LABEL: s.PLAYER,
            s.COURT[0]: xy[:, 0],
            s.COURT[1]: xy[:, 1],
        }
    )


def test_shake_is_removed_but_a_straight_run_is_kept() -> None:
    rng = np.random.default_rng(0)
    true = np.stack([np.linspace(0, 6, 60), np.full(60, 5.0)], axis=1)  # 3 m/s
    noisy = true + rng.normal(0, 0.15, true.shape)
    out = smooth_paths(_track(noisy))[s.COURT].to_numpy()
    assert np.abs(out - true).mean() < 0.5 * np.abs(noisy - true).mean()


def test_single_frame_spike_is_removed() -> None:
    xy = np.stack([np.linspace(0, 3, 30), np.full(30, 5.0)], axis=1)
    xy[15] += [4.0, 4.0]
    out = smooth_paths(_track(xy))[s.COURT].to_numpy()
    assert np.abs(out[15] - [1.55, 5.0]).max() < 0.3


def test_gap_starts_a_new_run_and_ball_is_untouched() -> None:
    xy = np.array([[0.0, 0.0]] * 10 + [[10.0, 10.0]] * 10)
    frames = np.r_[np.arange(10), np.arange(50, 60)]
    player = _track(xy, frames)
    ball = _track(xy, frames)
    ball[s.LABEL], ball[s.TRACK_ID] = s.BALL, -1
    tracks = pd.concat([player, ball], ignore_index=True)
    out = smooth_paths(tracks)
    assert np.allclose(out[s.COURT].to_numpy(), tracks[s.COURT].to_numpy())


def _box_rows(track_id: int, frames: range, x0: float, team: int = 0) -> list[dict[str, object]]:
    return [
        {
            s.FRAME: f,
            s.TRACK_ID: track_id,
            s.LABEL: s.PLAYER,
            s.TEAM: team,
            "x1": x0 + 2 * f,
            "y1": 0.0,
            "x2": x0 + 2 * f + 40,
            "y2": 100.0,
        }
        for f in frames
    ]


def test_short_gap_is_filled_with_an_interpolated_box() -> None:
    from cv_basketball.paths import fill_gaps

    people = pd.DataFrame(_box_rows(1, range(0, 10), 0) + _box_rows(1, range(15, 20), 0))
    out = fill_gaps(people, max_gap=30)
    filled = out[out[s.INTERPOLATED]]
    assert filled[s.FRAME].tolist() == [10, 11, 12, 13, 14]
    assert np.allclose(filled["x1"], 2 * filled[s.FRAME])
    assert (filled[s.TEAM] == 0).all()


def test_gap_is_not_filled_on_top_of_another_box() -> None:
    from cv_basketball.paths import fill_gaps

    # track 2 is a box right where track 1 would be during its gap
    people = pd.DataFrame(
        _box_rows(1, range(0, 10), 0)
        + _box_rows(1, range(15, 20), 0)
        + _box_rows(2, range(10, 15), 1)
    )
    assert not fill_gaps(people, max_gap=30)[s.INTERPOLATED].any()


def test_new_id_covering_a_gap_is_stitched_back() -> None:
    from cv_basketball.paths import stitch_into_gaps

    # track 1 loses the player for frames 10-19; track 2 is the same player meanwhile,
    # starting as a second box on them at frame 8
    people = pd.DataFrame(
        _box_rows(1, range(0, 10), 0)
        + _box_rows(1, range(20, 30), 0)
        + _box_rows(2, range(8, 20), 0)
    )
    out = stitch_into_gaps(people)
    assert set(out[s.TRACK_ID]) == {1}
    assert out[s.FRAME].is_unique  # the duplicate boxes at frames 8-9 are gone
    assert sorted(out[s.FRAME]) == list(range(30))
