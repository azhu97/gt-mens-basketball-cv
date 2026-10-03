"""Render the annotated output video from the tracks table (second pass over the video)."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cv_basketball import schema as s
from cv_basketball.court import CourtSpec, court_lines, court_to_pixel, draw_court
from cv_basketball.video import Frame, VideoInfo, iter_frames, open_writer, video_info

TEAM_COLORS: dict[int, tuple[int, int, int]] = {  # BGR
    0: (60, 60, 230),
    1: (230, 160, 40),
    -1: (160, 160, 160),
}
BALL_COLOR = (0, 220, 255)
TRAIL_LEN = 20
MINIMAP_PX_PER_M = 10.0
COURT_VIDEO_PX_PER_M = 40.0
COURT_LINE_COLOR = (255, 0, 255)


def _draw_court_dots(img: Frame, rows: pd.DataFrame, px_per_m: float) -> None:
    """Draw players (team-colored) and the ball at their court coordinates on a court image."""
    if s.COURT[0] not in rows:
        return
    rows = rows.dropna(subset=list(s.COURT))
    radius = max(2, round(px_per_m / 2))
    pts = court_to_pixel(rows[s.COURT].to_numpy(dtype=np.float64), px_per_m)
    for (px, py), label, team in zip(pts, rows[s.LABEL], rows[s.TEAM], strict=True):
        if label == s.BALL:
            cv2.circle(img, (int(px), int(py)), max(2, radius * 3 // 5), BALL_COLOR, -1)
        else:
            color = TEAM_COLORS.get(int(team), TEAM_COLORS[-1])
            cv2.circle(img, (int(px), int(py)), radius, color, -1)


def _draw_minimap(frame: Frame, rows: pd.DataFrame, court_img: Frame) -> None:
    mini = court_img.copy()
    _draw_court_dots(mini, rows, MINIMAP_PX_PER_M)
    mh, mw = mini.shape[:2]
    fh, fw = frame.shape[:2]
    if mh + 10 <= fh and mw + 10 <= fw:
        frame[10 : 10 + mh, fw - mw - 10 : fw - 10] = mini


def _draw_court_lines(
    frame: Frame, lines: list[NDArray[np.float64]], image_to_court: NDArray[np.float64]
) -> None:
    """Overlay the court model as the calibration sees it, to check the registration."""
    court_to_image = np.linalg.inv(image_to_court)
    for line in lines:
        pts = cv2.perspectiveTransform(line.reshape(-1, 1, 2), court_to_image)
        cv2.polylines(frame, [pts.astype(np.int32)], False, COURT_LINE_COLOR, 1)


def render_video(
    video: Path,
    tracks: pd.DataFrame,
    out_path: Path,
    court: CourtSpec | None = None,
    homographies: NDArray[np.float64] | None = None,
    minimap: bool = True,
) -> None:
    """Draw player boxes (team-colored, with IDs), the ball and its trail, and a minimap.

    The minimap is drawn only when ``court`` is given, ``minimap`` is set, and tracks have
    court coordinates. With per-frame image->court ``homographies`` too, the court lines
    are overlaid.
    """
    info = video_info(video)
    writer = open_writer(out_path, info)
    by_frame = dict(tuple(tracks.groupby(s.FRAME)))
    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LEN)
    court_img = draw_court(court, MINIMAP_PX_PER_M) if court is not None and minimap else None
    lines = court_lines(court) if court is not None else []
    empty = tracks.iloc[0:0]

    try:
        for idx, frame in iter_frames(video):
            rows = by_frame.get(idx, empty)
            if lines and homographies is not None and idx < len(homographies):
                _draw_court_lines(frame, lines, homographies[idx])
            for r in rows.itertuples(index=False):
                x1, y1, x2, y2 = (int(getattr(r, c)) for c in s.BOX)
                if r.label == s.BALL:
                    trail.append(((x1 + x2) // 2, (y1 + y2) // 2))
                    continue
                color = TEAM_COLORS.get(int(getattr(r, s.TEAM)), TEAM_COLORS[-1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, str(r.track_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )
            for i, (px, py) in enumerate(trail):
                cv2.circle(frame, (px, py), 2 + 4 * i // TRAIL_LEN, BALL_COLOR, -1)
            if court_img is not None:
                _draw_minimap(frame, rows, court_img)
            writer.write(frame)
    finally:
        writer.release()


def render_court_video(
    tracks: pd.DataFrame, out_path: Path, court: CourtSpec, fps: float, n_frames: int
) -> None:
    """Top-down court video with a dot per player and the ball, one frame per video frame.

    Drawn from the tracks table alone, so it never reads the source video.
    """
    court_img = draw_court(court, COURT_VIDEO_PX_PER_M)
    h, w = court_img.shape[:2]
    writer = open_writer(out_path, VideoInfo(fps=fps, width=w, height=h, n_frames=n_frames))
    by_frame = dict(tuple(tracks.groupby(s.FRAME)))
    try:
        for idx in range(n_frames):
            img = court_img.copy()
            if idx in by_frame:
                _draw_court_dots(img, by_frame[idx], COURT_VIDEO_PX_PER_M)
            writer.write(img)
    finally:
        writer.release()
