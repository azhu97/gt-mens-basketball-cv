"""`cvb` command-line interface."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from cv_basketball.court import COURTS

app = typer.Typer(no_args_is_help=True, help="Basketball video analysis with YOLO.")

DEVICES = ["mps", "cuda:0", "cpu"]
TRACKERS = ["bytetrack.yaml", "botsort.yaml"]  # ultralytics built-ins


# Shell completion for option values (flag names come from Typer); install with
# `cvb --install-completion`. These run on every <TAB>, so they only glob nearby files.


def _complete_paths(incomplete: str, suffix: str) -> list[str]:
    """Files ending in ``suffix`` (and directories to descend into) matching ``incomplete``."""
    matches = []
    for p in sorted(Path().glob(f"{incomplete}*")):
        if p.is_dir() and not p.name.startswith("."):
            matches.append(f"{p}/")
        elif p.name.endswith(suffix):
            matches.append(str(p))
    return matches


def _complete_model(incomplete: str) -> list[str]:
    """Trained weights under runs/train first, then any ``.pt`` path."""
    trained = [str(p) for p in sorted(Path("runs/train").glob("*/weights/best.pt"))]
    found = trained + [p for p in _complete_paths(incomplete, ".pt") if p not in trained]
    return [p for p in found if p.startswith(incomplete)]


def _complete_calibration(ctx: typer.Context, incomplete: str) -> list[str]:
    """The calibration saved next to the video being tracked, then any calibration JSON."""
    # during completion the positional video is often left unparsed in ctx.args
    video = ctx.params.get("video") or next(iter(ctx.args), None)
    sibling = [str(Path(video).with_suffix(".calibration.json"))] if video else []
    sibling = [p for p in sibling if Path(p).exists() and p.startswith(incomplete)]
    return sibling + [p for p in _complete_paths(incomplete, ".json") if p not in sibling]


def _complete_from(choices: list[str]) -> Callable[[str], list[str]]:
    return lambda incomplete: [c for c in choices if c.startswith(incomplete)]


@app.command()
def track(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[
        Path, typer.Option(help="Output root; results go to <out>/<video stem>/")
    ] = Path("runs"),
    model: Annotated[
        str,
        typer.Option(help="Ultralytics weights name or path", autocompletion=_complete_model),
    ] = "yolo11m.pt",
    device: Annotated[
        str | None,
        typer.Option(
            help="cuda:0 / mps / cpu (auto if omitted)", autocompletion=_complete_from(DEVICES)
        ),
    ] = None,
    tracker: Annotated[
        str | None,
        typer.Option(
            help="Tracker YAML (default: bundled basketball BoT-SORT; or bytetrack.yaml)",
            autocompletion=_complete_from(TRACKERS),
        ),
    ] = None,
    calibration: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            help="Calibration JSON from `cvb calibrate`",
            autocompletion=_complete_calibration,
        ),
    ] = None,
    person_conf: float = 0.4,
    ball_conf: float = 0.1,
    ball_imgsz: Annotated[int, typer.Option(help="Input size for the ball detector")] = 1280,
    csv: Annotated[bool, typer.Option(help="Also write tracks.csv")] = False,
    render: Annotated[bool, typer.Option(help="Write annotated.mp4")] = True,
    separate_court: Annotated[
        bool,
        typer.Option(help="Write the court view to its own court.mp4 instead of a minimap"),
    ] = False,
) -> None:
    """Detect and track players and ball, assign teams, and export results."""
    if separate_court and calibration is None:
        raise typer.BadParameter(
            "the court view needs court positions, which need --calibration "
            f"(make one with `cvb calibrate {video} --court ncaa --frame N`)",
            param_hint="--separate-court",
        )
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
        separate_court=separate_court,
    )
    out_dir = out / video.stem
    typer.echo(f"Tracking {video} on {cfg.device} -> {out_dir}")
    tracks = run(video, out_dir, cfg)
    typer.echo(f"Wrote {len(tracks)} rows to {out_dir}")


@app.command()
def flips(
    tracks: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="tracks.parquet from `cvb track`")
    ],
) -> None:
    """Report how stable the team labels in a tracks table are (see diagnostics.py)."""
    import pandas as pd

    from cv_basketball.diagnostics import flip_report

    for name, value in flip_report(pd.read_parquet(tracks)).as_dict().items():
        typer.echo(f"{name:30} {value}")


@app.command()
def calibrate(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option(help="Defaults to <video>.calibration.json")] = None,
    court: Annotated[
        str,
        typer.Option(
            help=f"One of: {', '.join(COURTS)}", autocompletion=_complete_from(list(COURTS))
        ),
    ] = "nba",
    frame: Annotated[int, typer.Option(help="Frame index to calibrate on")] = 0,
) -> None:
    """Click court landmarks on a frame to build an image-to-court homography."""
    from cv_basketball.calibrate import run_calibration

    if court not in COURTS:
        raise typer.BadParameter(f"Unknown court {court!r}; choose from {list(COURTS)}")
    out_path = out or video.with_suffix(".calibration.json")
    cal = run_calibration(video, COURTS[court], out_path, frame)
    typer.echo(f"Saved {len(cal.image_points)} points to {out_path}")


dataset_app = typer.Typer(no_args_is_help=True, help="Prepare fine-tuning datasets.")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("download")
def dataset_download(
    dataset: Annotated[str, typer.Argument(help="Roboflow Universe workspace/project/version")],
    out: Annotated[Path | None, typer.Option(help="Defaults to data/raw/<project>")] = None,
    api_key: Annotated[str, typer.Option(envvar="ROBOFLOW_API_KEY", help="Roboflow API key")] = "",
) -> None:
    """Download a Roboflow Universe dataset in YOLO format."""
    from cv_basketball.dataset import download_roboflow

    if not api_key:
        raise typer.BadParameter("Set ROBOFLOW_API_KEY or pass --api-key", param_hint="--api-key")
    dst = out or Path("data/raw") / dataset.split("/")[1]
    download_roboflow(dataset, dst, api_key)
    typer.echo(f"Downloaded {dataset} to {dst}")


@dataset_app.command("prepare")
def dataset_prepare(
    src: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="YOLO-format export")],
    out: Annotated[Path | None, typer.Option(help="Defaults to data/datasets/<src name>")] = None,
) -> None:
    """Remap a dataset's classes to player / referee / ball for fine-tuning."""
    from cv_basketball.dataset import remap_dataset

    dst = out or Path("data/datasets") / src.name
    counts = remap_dataset(src, dst)
    typer.echo(f"Wrote {dst / 'data.yaml'}; boxes per class: {dict(counts)}")


@app.command()
def train(
    data: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="data.yaml")],
    model: Annotated[
        str, typer.Option(help="Starting weights", autocompletion=_complete_model)
    ] = "yolo11m.pt",
    device: Annotated[
        str | None,
        typer.Option(
            help="cuda:0 / mps / cpu (auto if omitted)", autocompletion=_complete_from(DEVICES)
        ),
    ] = None,
    out: Annotated[Path, typer.Option(help="Training runs go to <out>/<name>/")] = Path(
        "runs/train"
    ),
    name: str = "basketball",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 4,
) -> None:
    """Fine-tune YOLO on a prepared dataset; use the printed best.pt with `cvb track --model`."""
    from cv_basketball.device import select_device
    from cv_basketball.train import train as run_train

    best = run_train(
        data,
        model=model,
        device=select_device(device),
        out=out,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )
    typer.echo(f"Best weights: {best}")
