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

uv run cvb calibrate VIDEO --court ncaa --frame N   # click landmarks; adds keyframe N to VIDEO.calibration.json
uv run cvb track VIDEO [--calibration X.json] [--device mps] [--csv] [--no-render] [--model best.pt]

uv run --env-file .env cvb dataset download WORKSPACE/PROJECT/VERSION   # key in .env; -> data/raw/PROJECT
uv run cvb dataset prepare data/raw/PROJECT             # remap classes -> data/datasets/PROJECT/data.yaml
uv run cvb train data/datasets/PROJECT/data.yaml        # -> runs/train/basketball*/weights/best.pt
```

## Architecture

This is an offline pipeline (`pipeline.run`) that makes **two passes over the video** and uses **one DataFrame as the contract between stages**:

1. **Pass 1, `detect_and_track`**: `tracking.track_video` reads frames with OpenCV and runs two YOLO instances on each one: `model.track(persist=True)` for players (bundled `botsort_basketball.yaml` by default) and a plain `predict` at `ball_imgsz=1280` for the ball. For each player box it samples the jersey color (`teams.torso_color_feature`, a LAB histogram of the upper-torso crop) right away, along with how far its feet are from the court floor (`floor.floor_hull`, found per frame because the camera pans; `floor_dist`). It also records the camera motion between consecutive frames (`camera.frame_motion`, a whole-image homography from optical flow, saved as `camera_motion.npy`). This way frames never need to be kept in memory.
2. **Post-process, `postprocess`**: this step works only on the DataFrame. Ball rows go through `track_ball`, which picks one candidate per frame as the best-scoring path through time: a ball must be able to travel between picks, and candidates near an on-court player score extra (confident decoys such as the mascot or rim break a per-frame pick). Then they go through `interpolate_ball`, a linear fill of short, physically reachable gaps, marked `interpolated=True`. People rows are first filtered by `floor.on_court_tracks`, which drops tracks whose feet are off the floor in most frames (fans, staff). Then they go through `labels.vote_person_labels` (one player/referee label per track). Player rows then go through `assign_teams`: KMeans on color, ignoring torsos covered by another box (`occluded_torsos`), then a majority vote per `track_id`, so a track never switches teams. Both steps first cut a track into fresh IDs wherever its label or color cluster changes for 30+ rows (`segments.split_on_change`). Those cuts are likely tracker ID swaps in collisions. If a calibration is given, `camera.court_homographies` carries its keyframes through the camera motion to give an image-to-court homography for every frame. A frame between two keyframes blends both predictions, because flow drifts. `homography.add_court_coords` then projects the bottom-centre of each box into court metres.
3. **Pass 2, `annotate.render_video`**: re-reads the video with OpenCV and draws from the DataFrame.

`schema.py` defines every column name in that DataFrame, and the table is also the exported `tracks.parquet`/`.csv` format. Use the `schema` constants rather than string literals. When you add a stage, add its columns there.

Other conventions:
- Model classes are mapped to labels **by class name** (`tracking.label_map`), so the same pipeline runs pretrained COCO weights (`person`, `sports ball`) and fine-tuned weights (`player`, `referee`, `ball`; see `dataset.TARGET_CLASSES`). Referees are tracked but get `team=-1` and are left out of team clustering. The ball gets `track_id=-1` because it's treated as a single object.
- Fine-tuning: `dataset.remap_dataset` collapses a YOLO-format export's fine-grained classes (`player-jump-shot`, `Ref`, `ball-in-basket`...) into those three by name (`target_label`) and drops the rest (rim, scoreboard...). Training defaults to imgsz 1280 for the ball, and the weights remember that size, so `track` runs players at 1280 with them too.
- The court coordinate system is in metres, with the origin at a baseline/sideline corner and x running along the length (`court.py`). A calibration JSON stores the court spec name (`nba`/`fiba`/`ncaa`; these clips are NCAA, with a 12 ft lane) and one or more **keyframes**. Each keyframe pairs image pixels with court metres at one frame. Run `cvb calibrate VIDEO --frame N` once per keyframe; it adds to the file. Use a keyframe near each end of a long pan, so both ends of the court are anchored.
- `device.select_device` picks CUDA, then MPS, then CPU. `torch` and `ultralytics` are imported lazily inside functions to keep the CLI and tests fast. Keep it that way.

## Gotchas

- The ball uses `predict`, not `track`, because `model.track` only returns boxes the tracker has confirmed and silently drops short-lived ball detections. It runs at imgsz 1280 because at 640 a 1080p ball is about 10px. It needs its own `YOLO` instance: calling `predict` on the tracking model would run its tracker callbacks.
- COCO `sports ball` gives many false positives on broadcast footage (crowd, pom-poms), and raising `ball_conf` doesn't separate them from the real ball.
- Ultralytics trackers need `lap`. If it's missing, ultralytics tries to `pip install` it at runtime, and that fails in a uv venv. It's declared in `pyproject.toml`, so keep it there.
- mypy treats `ultralytics` as untyped (`follow_imports = "skip"`) because its partial hints (e.g. `Boxes | None`) are noisy. cv2 *is* typed, so wrap cv2 return values in `np.asarray(..., dtype=...)` to satisfy the `NDArray` annotations.
- Team assignment always uses 2 clusters. With COCO weights there's no referee class, so refs and coaches get forced into a team. Fine-tuned weights with a `referee` class fix refs, but not coaches or bench players.
- Weights (`*.pt`), videos, `data/`, `runs/` and `.env` (holds `ROBOFLOW_API_KEY`) are gitignored. Never hardcode or commit the key. Ultralytics downloads weights into the current working directory.
