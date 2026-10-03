# cv-basketball

Offline basketball video analysis. It uses YOLO for detection with the ultralytics' BoT-SORT tracker to follow players, a separate high-resolution pass to detect the ball, clusters players into teams by jersey color, and maps positions onto a top-down court using a manual homography calibration.

## Setup

```bash
uv sync
uv run pre-commit install
```

Pretrained COCO weights (`yolo11m.pt` by default) download automatically on first run. The device is chosen automatically in the order CUDA, then Apple MPS, then CPU. Use `--device` to override it.

## Usage

```bash
# Optional, once per camera setup: click court landmarks on a frame
uv run cvb calibrate game.mp4 --court ncaa --frame 0     # writes game.calibration.json
uv run cvb calibrate game.mp4 --court ncaa --frame 400   # adds a second keyframe after a pan

# Track, assign teams, project to court, and render
uv run cvb track game.mp4 --calibration game.calibration.json --csv
```

Outputs go to `runs/<video stem>/`:

- `tracks.parquet` (and `tracks.csv` with `--csv`): one row per detection per frame. The columns are defined in `src/cv_basketball/schema.py`.
- `annotated.mp4`: team-colored player boxes with track IDs, the ball trail, and a court minimap when a calibration is given.
- `court.mp4` (with `--separate-court`): the court view as its own full-size top-down video instead of the minimap, frame-for-frame with `annotated.mp4`.

## Fine-tuning

The pretrained COCO weights have no referee class and a weak ball class. To fine-tune on a public basketball dataset from [Roboflow Universe](https://universe.roboflow.com) (needs a free API key):

```bash
echo 'ROBOFLOW_API_KEY=...' > .env                      # gitignored
uv run --env-file .env cvb dataset download WORKSPACE/PROJECT/VERSION   # YOLO-format export into data/raw/PROJECT
uv run cvb dataset prepare data/raw/PROJECT             # classes -> player / referee / ball
uv run cvb train data/datasets/PROJECT/data.yaml        # prints the path to best.pt
uv run cvb track game.mp4 --model runs/train/basketball/weights/best.pt
```

## Known limitations

- Both COCO and fine-tuned weights detect ball-like objects as the ball, sometimes with high confidence: the mascot's head, the rim, orange shoes, pom-poms. The ball is picked as a smooth path through time that favours candidates near a player, and short reachable gaps are interpolated. This is about 86% right on hand-labelled frames, but a confident decoy far from any player can still win while the ball is in the air. Fine-tuning with these decoys as negatives would fix it at the source.
- With COCO weights, referees are clustered into one of the two teams. Fine-tuned weights with a referee class fix this.
- Court coordinates need hand-clicked keyframes for each clip: at least one, and ideally one near each end of a long pan. Between keyframes the camera is followed with optical flow, which drifts slowly. Positions are of the feet, so they're noisy when feet are hidden, and the ball's position is only right when it's on the floor.
- Team colours still flicker briefly in places, usually just after a collision or when a new track ID starts. A new ID re-decides its team from a few noisy frames before settling. The fix is to carry a team across ID changes and to require a minimum number of clean frames before a track votes; this is deferred for now.
- Coaches and bench staff standing on the court surface along the sideline are kept and forced into a team. The off-court filter only removes people whose feet are off the floor, such as fans in the stands.
