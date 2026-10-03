"""End-to-end offline pipeline.

Pass 1 streams the video through YOLO (tracker for players, plain detection for the
ball), recording boxes, jersey colors and distance from the court floor (no frames kept
in memory). Post-processing (ball path through time, off-court people dropped, tracks split at
player/referee and team changes, per-track votes on both, court projection) then works
purely on the tracks DataFrame. Pass 1 also measures camera motion, which carries
hand-calibrated keyframes to every frame for court coordinates. Pass 2 re-reads the
video to render the annotated output.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s
from cv_basketball.annotate import render_court_video, render_video
from cv_basketball.ball import interpolate_ball, track_ball
from cv_basketball.ball2d import ground_ball
from cv_basketball.camera import court_homographies, frame_motion
from cv_basketball.floor import floor_distance, floor_hull, on_court_tracks
from cv_basketball.homography import (
    Calibration,
    add_court_coords,
    load_calibrations,
    reject_bad_keyframes,
)
from cv_basketball.labels import vote_person_labels
from cv_basketball.paths import smooth_paths
from cv_basketball.smoothing import drop_blips, smooth_teams
from cv_basketball.swaps import max_iou
from cv_basketball.teams import (
    UNKNOWN_TEAM,
    TeamParams,
    assign_teams,
    occluded_torsos,
    torso_color_feature,
)
from cv_basketball.tracking import DEFAULT_TRACKER, track_video
from cv_basketball.video import video_info


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
    separate_court: bool = False  # court view in its own court.mp4, not an inset
    teams: TeamParams = field(default_factory=TeamParams)
    # court-path smoothing windows in frames (see paths.py); 1 turns a filter off
    path_median: int = 5
    path_window: int = 15
    ground_ball: bool = True  # ball at its holder's feet / between holders (ball2d.py)


def detect_and_track(video: Path, cfg: PipelineConfig) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Pass 1: raw per-detection rows (see ``schema.DETECTION_COLUMNS``) and camera motion.

    The motion array has one homography per frame, mapping the previous frame's pixels to
    this frame's (identity for frame 0); see ``camera.frame_motion``.
    """
    rows: list[dict[str, Any]] = []
    motions: list[NDArray[np.float64]] = []
    prev_gray: NDArray[np.uint8] | None = None
    for fd in track_video(
        video,
        model_name=cfg.model,
        device=cfg.device,
        tracker=cfg.tracker,
        person_conf=cfg.person_conf,
        ball_conf=cfg.ball_conf,
        ball_imgsz=cfg.ball_imgsz,
    ):
        gray = np.asarray(cv2.cvtColor(fd.image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        people_boxes = fd.xyxy[[label != s.BALL for label in fd.labels]]
        motion = np.eye(3) if prev_gray is None else frame_motion(prev_gray, gray, people_boxes)
        motions.append(motion)
        prev_gray = gray
        hull = floor_hull(fd.image)
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
                    | {s.FLOOR_DIST: floor_distance(hull, box)}
                    | dict(zip(s.COLOR_FEATURES, map(float, color), strict=True))
                )
            elif label == s.BALL and conf >= cfg.ball_conf:
                rows.append({**base, s.TRACK_ID: -1, s.LABEL: s.BALL})
    return pd.DataFrame(rows, columns=s.DETECTION_COLUMNS), np.asarray(motions).reshape(-1, 3, 3)


def image_to_court(
    keyframes: list[Calibration], motions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-frame image->court homographies: keyframes carried through the camera motion."""
    return court_homographies(motions, keyframes)


def postprocess(
    raw: pd.DataFrame, cfg: PipelineConfig, homographies: NDArray[np.float64] | None
) -> pd.DataFrame:
    """Clean up pass-1 rows into the tracks table.

    ``homographies`` holds one image->court homography per frame (see ``image_to_court``);
    without it there are no court coordinates.
    """
    people = raw[raw[s.LABEL].isin([s.PLAYER, s.REFEREE])]
    people = people[on_court_tracks(people)].copy()
    people[s.TRACKER_ID] = people[s.TRACK_ID]
    people[s.MAX_IOU] = max_iou(people)
    tp = cfg.teams
    next_track_id = int(people[s.TRACK_ID].max()) + 1 if len(people) else 0
    people[[s.TRACK_ID, s.LABEL]] = vote_person_labels(
        people, next_track_id, tp.split_window, tp.split_min_run
    )

    players = people[people[s.LABEL] == s.PLAYER].copy()
    next_track_id = int(people[s.TRACK_ID].max()) + 1 if len(people) else 0
    occluded = occluded_torsos(people, tp.occluded_overlap)
    referees = people[people[s.LABEL] == s.REFEREE].copy()
    players[[s.TRACK_ID, s.TEAM, s.TEAM_VOTE]] = assign_teams(
        players, next_track_id, occluded, referees=referees if len(referees) else None, params=tp
    )
    players[s.TEAM] = smooth_teams(players, tp.max_flip_rows)
    players = players[drop_blips(players, tp.min_unknown_rows)]
    players[s.INTERPOLATED] = False

    referees[s.TEAM] = UNKNOWN_TEAM
    referees[s.TEAM_VOTE] = UNKNOWN_TEAM
    referees[s.INTERPOLATED] = False

    balls = interpolate_ball(track_ball(raw[raw[s.LABEL] == s.BALL], people), cfg.ball_max_gap)
    balls[s.TEAM] = UNKNOWN_TEAM
    balls[[s.TRACKER_ID, s.TEAM_VOTE]] = -1

    tracks = (
        pd.concat([players, referees, balls], ignore_index=True)
        .sort_values([s.FRAME, s.LABEL, s.TRACK_ID])
        .reset_index(drop=True)
    )
    if homographies is not None:
        tracks = add_court_coords(tracks, dict(enumerate(homographies)))
        tracks = smooth_paths(tracks, cfg.path_median, cfg.path_window)
        tracks = ground_ball(tracks) if cfg.ground_ball else tracks
    return tracks


def run(video: Path, out_dir: Path, cfg: PipelineConfig) -> pd.DataFrame:
    """Run the full pipeline, writing ``tracks.parquet`` (+ csv) and ``annotated.mp4``.

    With a calibration, the court view is a minimap inside ``annotated.mp4``, or its own
    ``court.mp4`` when ``cfg.separate_court`` is set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keyframes = load_calibrations(cfg.calibration) if cfg.calibration else None

    raw, motions = detect_and_track(video, cfg)
    np.save(out_dir / "camera_motion.npy", motions)
    if keyframes:
        keyframes, rejected = reject_bad_keyframes(keyframes, raw)
        for k in rejected:
            warnings.warn(
                f"Ignoring the calibration keyframe at frame {k.frame}: it maps most players "
                f"off the court. Re-click it with `cvb calibrate {video} --frame {k.frame}`.",
                stacklevel=1,
            )
    homographies = image_to_court(keyframes, motions) if keyframes else None
    tracks = postprocess(raw, cfg, homographies)
    tracks.to_parquet(out_dir / "tracks.parquet", index=False)
    if cfg.export_csv:
        tracks.to_csv(out_dir / "tracks.csv", index=False)
    if cfg.render:
        court = keyframes[0].spec if keyframes else None
        render_video(
            video,
            tracks,
            out_dir / "annotated.mp4",
            court,
            homographies,
            minimap=not cfg.separate_court,
        )
        if court is not None and cfg.separate_court:
            info = video_info(video)
            render_court_video(tracks, out_dir / "court.mp4", court, info.fps, len(motions))
    return tracks
