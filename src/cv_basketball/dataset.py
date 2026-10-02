"""Training data: download a Roboflow export and remap its classes to ours.

Public basketball datasets use many fine-grained classes (``player-jump-shot``, ``Ref``,
``ball-in-basket``, ``rim``, scoreboard fields...). Fine-tuning only needs the three the
pipeline uses, so ``remap_dataset`` collapses them into ``TARGET_CLASSES`` and drops the rest.
"""

import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import yaml

from cv_basketball import schema as s

TARGET_CLASSES = [s.PLAYER, s.REFEREE, s.BALL]
SPLITS = ("train", "valid", "val", "test")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def target_label(name: str) -> str | None:
    """Map a source dataset's class name to one of ``TARGET_CLASSES``, or None to drop it."""
    n = name.lower()
    if "ref" in n:
        return s.REFEREE
    if n == "person" or n.startswith("player"):
        return s.PLAYER
    if "ball" in n:
        return s.BALL
    return None


def download_roboflow(dataset: str, dst: Path, api_key: str, fmt: str = "yolov11") -> Path:
    """Download ``workspace/project/version`` from Roboflow Universe in YOLO format into ``dst``."""
    workspace, project, version = dataset.split("/")
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={api_key}"
    with urllib.request.urlopen(url) as resp:
        link = json.load(resp)["export"]["link"]
    dst.mkdir(parents=True, exist_ok=True)
    archive = dst / "export.zip"
    urllib.request.urlretrieve(link, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dst)
    archive.unlink()
    return dst


def remap_dataset(src: Path, dst: Path) -> Counter[str]:
    """Copy a YOLO-format dataset from ``src`` to ``dst`` with classes remapped.

    ``src`` must contain ``data.yaml`` (with ``names``) and split folders such as
    ``train/images`` + ``train/labels``. Writes ``dst/data.yaml`` and returns box counts
    per target class. Images whose boxes are all dropped are kept as negatives.
    """
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    mapping = {
        int(i): TARGET_CLASSES.index(lab)
        for i, n in names.items()
        if (lab := target_label(n)) is not None
    }

    counts: Counter[str] = Counter()
    splits: dict[str, str] = {}
    for split in SPLITS:
        images = src / split / "images"
        if not images.is_dir():
            continue
        out_images, out_labels = dst / split / "images", dst / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        for img in images.iterdir():
            if img.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            shutil.copy2(img, out_images / img.name)
            label_file = src / split / "labels" / f"{img.stem}.txt"
            lines = label_file.read_text().splitlines() if label_file.exists() else []
            kept = []
            for line in lines:
                cls, *coords = line.split()
                if int(cls) in mapping:
                    new = mapping[int(cls)]
                    kept.append(" ".join([str(new), *coords]))
                    counts[TARGET_CLASSES[new]] += 1
            (out_labels / f"{img.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        splits["val" if split == "valid" else split] = f"{split}/images"

    if "train" not in splits or "val" not in splits:
        raise ValueError(f"{src} needs train/ and valid/ (or val/) split folders")
    data = {"path": str(dst.resolve()), **splits, "names": dict(enumerate(TARGET_CLASSES))}
    (dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return counts
