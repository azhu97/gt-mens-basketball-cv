"""Interactive OpenCV tool to click court landmarks on a frame and save a Calibration."""

from pathlib import Path

import cv2

from cv_basketball.court import CourtSpec
from cv_basketball.homography import MIN_POINTS, Calibration
from cv_basketball.video import read_frame

_WINDOW = "cvb calibrate"


def run_calibration(video: Path, spec: CourtSpec, out: Path, frame_idx: int = 0) -> Calibration:
    """Prompt for each named landmark in turn.

    Left-click to place it, ``s`` to skip it (not visible), ``u`` to undo the last
    point, ``q`` / Esc to finish. At least four non-collinear points are required.
    """
    frame = read_frame(video, frame_idx)
    landmarks = list(spec.landmarks().items())
    image_points: list[tuple[float, float]] = []
    court_points: list[tuple[float, float]] = []
    clicked: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    i = 0
    try:
        while i < len(landmarks):
            name, court_xy = landmarks[i]
            canvas = frame.copy()
            for px, py in image_points:
                cv2.circle(canvas, (int(px), int(py)), 6, (0, 0, 255), -1)
            prompt = f"[{i + 1}/{len(landmarks)}] click: {name}  (s=skip u=undo q=done)"
            cv2.putText(canvas, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow(_WINDOW, canvas)
            key = cv2.waitKey(30) & 0xFF

            if clicked:
                image_points.append(clicked.pop())
                court_points.append(court_xy)
                clicked.clear()
                i += 1
            elif key == ord("s"):
                i += 1
            elif key == ord("u") and image_points:
                image_points.pop()
                court_points.pop()
                i = max(i - 1, 0)
            elif key in (ord("q"), 27):
                break
    finally:
        cv2.destroyWindow(_WINDOW)

    if len(image_points) < MIN_POINTS:
        raise ValueError(f"Only {len(image_points)} points placed; need at least {MIN_POINTS}")
    cal = Calibration(court=spec.name, image_points=image_points, court_points=court_points)
    cal.homography()  # validate before saving
    cal.save(out)
    return cal
