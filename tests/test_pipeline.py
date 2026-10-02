from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cv_basketball import schema as s
from cv_basketball.annotate import render_video
from cv_basketball.cli import app
from cv_basketball.court import NBA
from cv_basketball.pipeline import PipelineConfig, postprocess, run
from cv_basketball.video import video_info


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
            "x1": 0.0,
            "y1": 0.0,
            "x2": 10.0,
            "y2": 30.0,
            "torso_l": lum,
            "torso_a": 128.0,
            "torso_b": 128.0,
        }
        for f in range(3)
        for tid, label, lum in [(1, s.PLAYER, 30.0), (2, s.PLAYER, 220.0), (3, s.REFEREE, 120.0)]
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
                "torso_l": 50.0,
                "torso_a": 150.0,
                "torso_b": 150.0,
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
    tracks = postprocess(raw, PipelineConfig(), calibration=None)
    assert set(tracks[s.LABEL]) == {s.PLAYER, s.BALL}
    assert {s.TEAM, s.INTERPOLATED} <= set(tracks.columns)

    out = tmp_path / "annotated.mp4"
    render_video(synthetic_video, tracks, out, court=NBA)
    assert video_info(out).n_frames == 10


@pytest.mark.slow
def test_full_pipeline_with_yolo(synthetic_video: Path, tmp_path: Path) -> None:
    """Downloads yolo11n.pt on first run."""
    tracks = run(synthetic_video, tmp_path, PipelineConfig(model="yolo11n.pt", device="cpu"))
    assert (tmp_path / "tracks.parquet").exists()
    assert (tmp_path / "annotated.mp4").exists()
    assert set(s.DETECTION_COLUMNS) <= set(tracks.columns)
