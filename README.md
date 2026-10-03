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
uv run cvb calibrate game.mp4 --court nba          # writes game.calibration.json

# Track, assign teams, project to court, and render
uv run cvb track game.mp4 --calibration game.calibration.json --csv
```

Outputs go to `runs/<video stem>/`:

- `tracks.parquet` (and `tracks.csv` with `--csv`): one row per detection per frame. The columns are defined in `src/cv_basketball/schema.py`.
- `annotated.mp4`: team-colored player boxes with track IDs, the ball trail, and a court minimap when a calibration is given.

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

- The pretrained COCO `sports ball` class has many false positives on broadcast footage (e.g. pom-poms in the crowd). The highest-confidence detection per frame is kept, and short gaps are filled by interpolation.
- With COCO weights, referees are clustered into one of the two teams. Fine-tuned weights with a referee class fix this.
- Calibration assumes a static camera.
- Team colours still flicker briefly in places, usually just after a collision or when a new track ID starts. A new ID re-decides its team from a few noisy frames before settling. The fix is to carry a team across ID changes and to require a minimum number of clean frames before a track votes; this is deferred for now.
- Coaches and bench staff standing on the court surface along the sideline are kept and forced into a team. The off-court filter only removes people whose feet are off the floor, such as fans in the stands.
