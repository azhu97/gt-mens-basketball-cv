# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses uv on Python 3.12. Always run tools through `uv run`.

```bash
uv sync                                   # install deps + dev group
uv run pytest -m "not slow"               # fast tests (no model weights needed)
uv run pytest                             # all tests; slow ones download yolo11n.pt
uv run pytest tests/test_ball.py::test_interpolate_fills_short_gap_linearly   # single test
uv run ruff check --fix . && uv run ruff format .
uv run mypy                               # strict; checks src/ and tests/
uv run pre-commit run --all-files         # ruff + ruff-format + mypy

uv run cvb calibrate VIDEO --court nba    # interactive OpenCV window; writes VIDEO.calibration.json
uv run cvb track VIDEO [--calibration X.json] [--device mps] [--csv] [--no-render]
```

## Architecture

This is an offline pipeline (`pipeline.run`) that makes **two passes over the video** and uses **one DataFrame as the contract between stages**:

1. **Pass 1, `detect_and_track`**: `tracking.track_video` reads frames with OpenCV and runs two YOLO instances on each one: `model.track(persist=True)` for players (bundled `botsort_basketball.yaml` by default) and a plain `predict` at `ball_imgsz=1280` for the ball. For each player box it samples the jersey color (`teams.torso_color_feature`, mean LAB of the upper-torso crop) right away, so frames never need to be kept in memory.
2. **Post-process, `postprocess`**: this step works only on the DataFrame. Ball rows go through `select_ball` (the top-confidence detection per frame) and then `interpolate_ball` (linear fill of short gaps, marked `interpolated=True`). Player rows go through `assign_teams` (KMeans on color, then a majority vote per `track_id`, so a track never switches teams). If a calibration is given, `homography.add_court_coords` projects the bottom-centre of each box into court metres.
3. **Pass 2, `annotate.render_video`**: re-reads the video with OpenCV and draws from the DataFrame.

`schema.py` defines every column name in that DataFrame, and the table is also the exported `tracks.parquet`/`.csv` format. Use the `schema` constants rather than string literals. When you add a stage, add its columns there.

Other conventions:
- Detection uses pretrained COCO weights: class `PERSON=0` becomes players and `SPORTS_BALL=32` becomes the ball. The ball gets `track_id=-1` because it's treated as a single object and tracker IDs are ignored.
- The court coordinate system is in metres, with the origin at a baseline/sideline corner and x running along the length (`court.py`). A calibration JSON stores pairs of image pixels and court metres plus a court spec name (`nba`/`fiba`), and assumes a **static camera**.
- `device.select_device` picks CUDA, then MPS, then CPU. `torch` and `ultralytics` are imported lazily inside functions to keep the CLI and tests fast. Keep it that way.

## Gotchas

- The ball uses `predict`, not `track`, because `model.track` only returns boxes the tracker has confirmed and silently drops short-lived ball detections. It runs at imgsz 1280 because at 640 a 1080p ball is about 10px. It needs its own `YOLO` instance: calling `predict` on the tracking model would run its tracker callbacks.
- COCO `sports ball` gives many false positives on broadcast footage (crowd, pom-poms), and raising `ball_conf` doesn't separate them from the real ball.
- Ultralytics trackers need `lap`. If it's missing, ultralytics tries to `pip install` it at runtime, and that fails in a uv venv. It's declared in `pyproject.toml`, so keep it there.
- mypy treats `ultralytics` as untyped (`follow_imports = "skip"`) because its partial hints (e.g. `Boxes | None`) are noisy. cv2 *is* typed, so wrap cv2 return values in `np.asarray(..., dtype=...)` to satisfy the `NDArray` annotations.
- Team assignment always uses 2 clusters, so referees get forced into a team.
- Weights (`*.pt`), videos, `data/` and `runs/` are gitignored. Ultralytics downloads weights into the current working directory.
