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

## Known limitations

- The pretrained COCO `sports ball` class has many false positives on broadcast footage (e.g. pom-poms in the crowd). The highest-confidence detection per frame is kept, and short gaps are filled by interpolation.
- Referees are clustered into one of the two teams.
- Calibration assumes a static camera.
