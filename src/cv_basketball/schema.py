"""Column contract for the tracks table (one row per detection per frame).

Every stage reads/writes this single DataFrame; it is also the exported file format.
"""

FRAME = "frame"
TRACK_ID = "track_id"  # tracker ID for players; -1 for the ball (single object, ID unused)
LABEL = "label"  # "player" | "ball"
CONF = "conf"  # NaN for interpolated ball rows
BOX = ["x1", "y1", "x2", "y2"]  # pixel coords
COLOR_FEATURES = ["torso_l", "torso_a", "torso_b"]  # mean LAB jersey color (players only)
TEAM = "team"  # 0/1 for players, -1 for unknown / ball
INTERPOLATED = "interpolated"  # True for ball rows filled between detections
COURT = ["court_x", "court_y"]  # metres on the top-down court; only with a calibration

PLAYER = "player"
BALL = "ball"

DETECTION_COLUMNS = [FRAME, TRACK_ID, LABEL, CONF, *BOX, *COLOR_FEATURES]
