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
# Upper-torso LAB histogram (people only): 16 L bins, then 8 a and 8 b bins, each block a
# share of the crop's pixels. A mean color would average white fabric and dark skin or
# numbers into mid-grey; the histogram keeps the white jersey's share of bright pixels.
HIST_BINS_L, HIST_BINS_AB = 16, 8
COLOR_FEATURES = [f"torso_hist_{i}" for i in range(HIST_BINS_L + 2 * HIST_BINS_AB)]
# Signed distance of a person's feet from the bright court floor, in box heights: positive
# inside, negative outside (see floor.py). Off-court tracks are dropped in postprocess.
FLOOR_DIST = "floor_dist"
TEAM = "team"  # 0/1 for players, -1 for unknown / referee / ball
INTERPOLATED = "interpolated"  # True for ball rows filled between detections
COURT = ["court_x", "court_y"]  # metres on the top-down court; only with a calibration

PLAYER = "player"
REFEREE = "referee"  # only from fine-tuned weights with a referee class
BALL = "ball"

DETECTION_COLUMNS = [FRAME, TRACK_ID, LABEL, CONF, *BOX, FLOOR_DIST, *COLOR_FEATURES]
