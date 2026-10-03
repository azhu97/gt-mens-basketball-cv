"""End-to-end offline pipeline.

Pass 1 streams the video through YOLO (tracker for players, plain detection for the
ball), recording boxes and jersey colors (no frames kept in memory). Post-processing
(ball cleanup, tracks split at player/referee and team changes, per-track votes on
both, court projection) then works purely on the tracks DataFrame. Pass 2
re-reads the video to render the annotated output.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cv_basketball import schema as s
from cv_basketball.annotate import render_video
from cv_basketball.ball import interpolate_ball, select_ball
from cv_basketball.homography import Calibration, add_court_coords
from cv_basketball.labels import vote_person_labels
from cv_basketball.teams import (
    UNKNOWN_TEAM,
    assign_teams,
    occluded_torsos,
    torso_color_feature,
)
from cv_basketball.tracking import DEFAULT_TRACKER, track_video


@dataclass
class PipelineConfig:
    model: str = "yolo11m.pt"
    device: str = "cpu"
    tracker: str = DEFAULT_TRACKER
    person_conf: float = 0.4
    ball_conf: float = 0.1
    ball_imgsz: int = 1280
    ball_max_gap: int = 10
    calibration: Path | None = None
    export_csv: bool = False
    render: bool = True


def detect_and_track(video: Path, cfg: PipelineConfig) -> pd.DataFrame:
    """Pass 1: raw per-detection rows (see ``schema.DETECTION_COLUMNS``)."""
    rows: list[dict[str, Any]] = []
    for fd in track_video(
        video,
        model_name=cfg.model,
        device=cfg.device,
        tracker=cfg.tracker,
        person_conf=cfg.person_conf,
        ball_conf=cfg.ball_conf,
        ball_imgsz=cfg.ball_imgsz,
    ):
        for box, tid, label, conf in zip(fd.xyxy, fd.track_ids, fd.labels, fd.confs, strict=True):
            base = {
                s.FRAME: fd.frame_idx,
                s.CONF: float(conf),
                **dict(zip(s.BOX, map(float, box), strict=True)),
            }
            if label in (s.PLAYER, s.REFEREE) and conf >= cfg.person_conf:
                color = torso_color_feature(fd.image, box)
                rows.append(
                    {**base, s.TRACK_ID: int(tid), s.LABEL: label}
                    | dict(zip(s.COLOR_FEATURES, map(float, color), strict=True))
                )
            elif label == s.BALL and conf >= cfg.ball_conf:
                rows.append({**base, s.TRACK_ID: -1, s.LABEL: s.BALL})
    return pd.DataFrame(rows, columns=s.DETECTION_COLUMNS)


def postprocess(
    raw: pd.DataFrame, cfg: PipelineConfig, calibration: Calibration | None
) -> pd.DataFrame:
    people = raw[raw[s.LABEL].isin([s.PLAYER, s.REFEREE])].copy()
    next_track_id = int(people[s.TRACK_ID].max()) + 1 if len(people) else 0
    people[[s.TRACK_ID, s.LABEL]] = vote_person_labels(people, next_track_id)

    players = people[people[s.LABEL] == s.PLAYER].copy()
    next_track_id = int(people[s.TRACK_ID].max()) + 1 if len(people) else 0
    occluded = occluded_torsos(people)
    players[[s.TRACK_ID, s.TEAM]] = assign_teams(players, next_track_id, occluded)
    players[s.INTERPOLATED] = False

    referees = people[people[s.LABEL] == s.REFEREE].copy()
    referees[s.TEAM] = UNKNOWN_TEAM
    referees[s.INTERPOLATED] = False

    balls = interpolate_ball(select_ball(raw[raw[s.LABEL] == s.BALL]), cfg.ball_max_gap)
    balls[s.TEAM] = UNKNOWN_TEAM

    tracks = (
        pd.concat([players, referees, balls], ignore_index=True)
        .sort_values([s.FRAME, s.LABEL, s.TRACK_ID])
        .reset_index(drop=True)
    )
    if calibration is not None:
        tracks = add_court_coords(tracks, calibration.homography())
    return tracks


def run(video: Path, out_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    """Run the full pipeline, writing ``tracks.parquet`` (+ csv) and ``annotated.mp4``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = Calibration.load(cfg.calibration) if cfg.calibration else None

    tracks = postprocess(detect_and_track(video, cfg), cfg, calibration)
    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    if cfg.export_csv:
        tracks.to_csv(out_dir / "tracks.csv", index=False)
    if cfg.render:
        court = calibration.spec if calibration else None
        render_video(video, tracks, out_dir / "annotated.mp4", court)
    return tracks
