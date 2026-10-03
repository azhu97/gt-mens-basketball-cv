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
