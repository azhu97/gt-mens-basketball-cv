from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cv_basketball import schema as s
from cv_basketball.annotate import render_court_video, render_video
from cv_basketball.cli import app
from cv_basketball.court import NBA
from cv_basketball.pipeline import PipelineConfig, postprocess, run
from cv_basketball.teams import torso_color_feature
from cv_basketball.video import video_info


def _jersey(grey: int) -> dict[str, float]:
    """Color feature columns for a uniformly grey jersey."""
    frame = np.full((40, 20, 3), grey, dtype=np.uint8)
    feat = torso_color_feature(frame, np.array([0, 0, 20, 40], dtype=np.float32))
    return dict(zip(s.COLOR_FEATURES, map(float, feat), strict=True))


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "track" in result.stdout
    assert "calibrate" in result.stdout


def test_referees_get_no_team() -> None:
    rows = [
        {
            s.FRAME: f,
            s.TRACK_ID: tid,
            s.LABEL: label,
            s.CONF: 0.9,
            "x1": 20.0 * tid,
            "y1": 0.0,
            "x2": 20.0 * tid + 10.0,
            "y2": 30.0,
            **_jersey(grey),
        }
        for f in range(3)
        for tid, label, grey in [(1, s.PLAYER, 30), (2, s.PLAYER, 220), (3, s.REFEREE, 120)]
    ]
    tracks = postprocess(pd.DataFrame(rows, columns=s.DETECTION_COLUMNS), PipelineConfig(), None)
    teams = tracks.groupby(s.TRACK_ID)[s.TEAM].first()
    assert teams[3] == -1
    assert {teams[1], teams[2]} == {0, 1}


def test_postprocess_and_render_without_model(synthetic_video: Path, tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                s.FRAME: f,
                s.TRACK_ID: 1,
                s.LABEL: s.PLAYER,
                s.CONF: 0.9,
                "x1": 20.0 + 10 * f,
                "y1": 60.0,
                "x2": 60.0 + 10 * f,
                "y2": 180.0,
                **_jersey(50),
            }
            for f in range(10)
        ]
        + [
            {
                s.FRAME: 0,
                s.TRACK_ID: -1,
                s.LABEL: s.BALL,
                s.CONF: 0.5,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 8.0,
                "y2": 8.0,
            }
        ],
        columns=s.DETECTION_COLUMNS,
    )
    tracks = postprocess(raw, PipelineConfig(), homographies=None)
    assert set(tracks[s.LABEL]) == {s.PLAYER, s.BALL}
    assert {s.TEAM, s.INTERPOLATED} <= set(tracks.columns)

    out = tmp_path / "annotated.mp4"
    render_video(synthetic_video, tracks, out, court=NBA)
    assert video_info(out).n_frames == 10


def test_court_video_has_a_frame_per_video_frame(tmp_path: Path) -> None:
    tracks = pd.DataFrame(
        {
            s.FRAME: [0, 2],
            s.LABEL: [s.PLAYER, s.BALL],
            s.TEAM: [0, -1],
            s.COURT[0]: [5.0, float("nan")],
            s.COURT[1]: [7.0, float("nan")],
        }
    )
    out = tmp_path / "court.mp4"
    render_court_video(tracks, out, NBA, fps=30.0, n_frames=5)
    info = video_info(out)
    assert info.n_frames == 5
    assert info.width > 1000  # full-size, not minimap scale


@pytest.mark.slow
def test_full_pipeline_with_yolo(synthetic_video: Path, tmp_path: Path) -> None:
    """Downloads yolo11n.pt on first run."""
    tracks = run(synthetic_video, tmp_path, PipelineConfig(model="yolo11n.pt", device="cpu"))
    assert (tmp_path / "tracks.parquet").exists()
    assert (tmp_path / "annotated.mp4").exists()
    assert set(s.DETECTION_COLUMNS) <= set(tracks.columns)
