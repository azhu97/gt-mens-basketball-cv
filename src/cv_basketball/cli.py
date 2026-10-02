"""`cvb` command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from cv_basketball.court import COURTS

app = typer.Typer(no_args_is_help=True, help="Basketball video analysis with YOLO.")


@app.command()
def track(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[
        Path, typer.Option(help="Output root; results go to <out>/<video stem>/")
    ] = Path("runs"),
    model: Annotated[str, typer.Option(help="Ultralytics weights name or path")] = "yolo11m.pt",
    device: Annotated[str | None, typer.Option(help="cuda:0 / mps / cpu (auto if omitted)")] = None,
    tracker: Annotated[
        str | None,
        typer.Option(help="Tracker YAML (default: bundled basketball BoT-SORT; or bytetrack.yaml)"),
    ] = None,
    calibration: Annotated[
        Path | None, typer.Option(exists=True, help="Calibration JSON from `cvb calibrate`")
    ] = None,
    person_conf: float = 0.4,
    ball_conf: float = 0.1,
    ball_imgsz: Annotated[int, typer.Option(help="Input size for the ball detector")] = 1280,
    csv: Annotated[bool, typer.Option(help="Also write tracks.csv")] = False,
    render: Annotated[bool, typer.Option(help="Write annotated.mp4")] = True,
) -> None:
    """Detect and track players and ball, assign teams, and export results."""
    from cv_basketball.device import select_device
    from cv_basketball.pipeline import PipelineConfig, run
    from cv_basketball.tracking import DEFAULT_TRACKER

    cfg = PipelineConfig(
        model=model,
        device=select_device(device),
        tracker=tracker or DEFAULT_TRACKER,
        person_conf=person_conf,
        ball_conf=ball_conf,
        ball_imgsz=ball_imgsz,
        calibration=calibration,
        export_csv=csv,
        render=render,
    )
    out_dir = out / video.stem
    typer.echo(f"Tracking {video} on {cfg.device} -> {out_dir}")
    tracks = run(video, out_dir, cfg)
    typer.echo(f"Wrote {len(tracks)} rows to {out_dir}")


@app.command()
def calibrate(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option(help="Defaults to <video>.calibration.json")] = None,
    court: Annotated[str, typer.Option(help=f"One of: {', '.join(COURTS)}")] = "nba",
    frame: Annotated[int, typer.Option(help="Frame index to calibrate on")] = 0,
) -> None:
    """Click court landmarks on a frame to build an image-to-court homography."""
    from cv_basketball.calibrate import run_calibration

    if court not in COURTS:
        raise typer.BadParameter(f"Unknown court {court!r}; choose from {list(COURTS)}")
    out_path = out or video.with_suffix(".calibration.json")
    cal = run_calibration(video, COURTS[court], out_path, frame)
    typer.echo(f"Saved {len(cal.image_points)} points to {out_path}")
