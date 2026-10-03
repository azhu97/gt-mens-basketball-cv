"""Column contract for the tracks table (one row per detection per frame).

Every stage reads/writes this single DataFrame; it is also the exported file format.
"""

FRAME = "frame"
# Tracker ID for people, split into fresh IDs where a track changes label or team (a
# likely ID swap; see segments.py); -1 for the ball (single object, ID unused)
TRACK_ID = "track_id"
LABEL = "label"  # "player" | "referee" | "ball"
CONF = "conf"  # NaN for interpolated ball rows
BOX = ["x1", "y1", "x2", "y2"]  # pixel coords
COLOR_FEATURES = ["torso_l", "torso_a", "torso_b"]  # mean LAB jersey color (people only)
TEAM = "team"  # 0/1 for players, -1 for unknown / referee / ball
INTERPOLATED = "interpolated"  # True for ball rows filled between detections
COURT = ["court_x", "court_y"]  # metres on the top-down court; only with a calibration

PLAYER = "player"
REFEREE = "referee"  # only from fine-tuned weights with a referee class
BALL = "ball"

DETECTION_COLUMNS = [FRAME, TRACK_ID, LABEL, CONF, *BOX, *COLOR_FEATURES]
