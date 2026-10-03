import cv2
import numpy as np
import pytest

from cv_basketball.camera import court_homographies, frame_motion
from cv_basketball.homography import Calibration, project


def _texture() -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (540, 960), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(noise, (5, 5), 0), dtype=np.uint8)


def test_frame_motion_recovers_a_pan() -> None:
    img = _texture()
    shift = np.array([[1, 0, 12], [0, 1, -7], [0, 0, 1]], dtype=np.float64)
    moved = np.asarray(cv2.warpPerspective(img, shift, (960, 540)), dtype=np.uint8)
    no_people = np.empty((0, 4), dtype=np.float32)
    M = frame_motion(img, moved, no_people)
    centre = np.array([[480.0, 270.0]])
    assert project(M, centre) == pytest.approx(centre + np.array([[12, -7]]), abs=0.5)


def _keyframe(frame: int, px_per_m: float, offset_x: float) -> Calibration:
    court = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    image = [(x * px_per_m + offset_x, y * px_per_m) for x, y in court]
    return Calibration("ncaa", image, court, frame=frame)


def test_static_camera_single_keyframe_applies_everywhere() -> None:
    motions = np.stack([np.eye(3)] * 5)
    Hs = court_homographies(motions, [_keyframe(2, 20, 0)])
    for H in Hs:
        assert project(H, np.array([[200.0, 100.0]])) == pytest.approx(np.array([[10.0, 5.0]]))


def test_keyframe_is_carried_through_camera_motion() -> None:
    # The camera pans 10 px right per frame, so scene content moves 10 px left.
    pan = np.array([[1, 0, -10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    motions = np.stack([np.eye(3)] + [pan] * 4)
    Hs = court_homographies(motions, [_keyframe(0, 20, 0)])
    # The court point imaged at (200, 100) in frame 0 is at (160, 100) in frame 4.
    court_xy = project(Hs[4], np.array([[160.0, 100.0]]))
    assert court_xy == pytest.approx(np.array([[10.0, 5.0]]), abs=1e-6)


def test_frames_between_keyframes_blend_towards_the_nearer_one() -> None:
    # Two inconsistent keyframes (flow drift): halfway, the court point (10, 0) lands
    # halfway between where each keyframe puts it.
    motions = np.stack([np.eye(3)] * 11)
    Hs = court_homographies(motions, [_keyframe(0, 20, 0), _keyframe(10, 20, 40)])
    mid_px = project(np.linalg.inv(Hs[5]), np.array([[10.0, 0.0]]))
    assert mid_px == pytest.approx(np.array([[220.0, 0.0]]), abs=1e-3)
    end_px = project(np.linalg.inv(Hs[10]), np.array([[10.0, 0.0]]))
    assert end_px == pytest.approx(np.array([[240.0, 0.0]]), abs=1e-3)
