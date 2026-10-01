"""Render the annotated output video from the tracks table (second pass over the video)."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cv_basketball import schema as s
from cv_basketball.court import CourtSpec, court_to_pixel, draw_court
from cv_basketball.video import Frame, iter_frames, open_writer, video_info

TEAM_COLORS: dict[int, tuple[int, int, int]] = {  # BGR
    0: (60, 60, 230),
    1: (230, 160, 40),
    -1: (160, 160, 160),
}
BALL_COLOR = (0, 220, 255)
TRAIL_LEN = 20
MINIMAP_PX_PER_M = 10.0


def _draw_minimap(frame: Frame, rows: pd.DataFrame, court_img: Frame) -> None:
    mini = court_img.copy()
    if s.COURT[0] in rows:
        pts = court_to_pixel(rows[s.COURT].to_numpy(dtype=np.float64), MINIMAP_PX_PER_M)
        for (px, py), label, team in zip(pts, rows[s.LABEL], rows[s.TEAM], strict=True):
            if label == s.BALL:
                cv2.circle(mini, (int(px), int(py)), 3, BALL_COLOR, -1)
            else:
                cv2.circle(
                    mini, (int(px), int(py)), 5, TEAM_COLORS.get(int(team), TEAM_COLORS[-1]), -1
                )
    mh, mw = mini.shape[:2]
    fh, fw = frame.shape[:2]
    if mh + 10 <= fh and mw + 10 <= fw:
        frame[10 : 10 + mh, fw - mw - 10 : fw - 10] = mini


def render_video(
    video: Path, tracks: pd.DataFrame, out_path: Path, court: CourtSpec | None = None
) -> None:
    """Draw player boxes (team-colored, with IDs), the ball and its trail, and a minimap.

    The minimap is drawn only when ``court`` is given and tracks have court coordinates.
    """
    info = video_info(video)
    writer = open_writer(out_path, info)
    by_frame = dict(tuple(tracks.groupby(s.FRAME)))
    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LEN)
    court_img = draw_court(court, MINIMAP_PX_PER_M) if court is not None else None
    empty = tracks.iloc[0:0]

    try:
        for idx, frame in iter_frames(video):
            rows = by_frame.get(idx, empty)
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
