"""Fine-tune YOLO weights on a dataset prepared by ``dataset.remap_dataset``."""

from pathlib import Path


def train(
    data: Path,
    *,
    model: str,
    device: str,
    out: Path,
    name: str,
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 4,
    patience: int = 15,
) -> Path:
    """Train from ``model`` (pretrained weights) and return the path to ``best.pt``.

    ``imgsz`` defaults to 1280 so the ball keeps enough pixels; the resulting weights
    remember it, so ``cvb track`` runs players at that size too. ``batch=4`` fits yolo11m
    at 1280 in ~18 GB on MPS; batch 8 exhausted a 24 GB Mac.
    """
    from ultralytics import YOLO  # lazy: heavy import

    yolo = YOLO(model)
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out.resolve()),
        name=name,
        patience=patience,
    )
    return Path(yolo.trainer.save_dir) / "weights" / "best.pt"
