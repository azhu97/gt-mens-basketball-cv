from pathlib import Path

import pytest
import yaml

from cv_basketball import schema as s
from cv_basketball.dataset import TARGET_CLASSES, remap_dataset, target_label
from cv_basketball.tracking import label_map


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("player", s.PLAYER),
        ("player-jump-shot", s.PLAYER),
        ("Player", s.PLAYER),
        ("Ref", s.REFEREE),
        ("referee", s.REFEREE),
        ("ball", s.BALL),
        ("Ball", s.BALL),
        ("ball-in-basket", s.BALL),
        ("rim", None),
        ("number", None),
        ("Shot Clock", None),
    ],
)
def test_target_label(name: str, expected: str | None) -> None:
    assert target_label(name) == expected


def test_label_map_handles_coco_and_finetuned_names() -> None:
    assert label_map({0: "person", 1: "bicycle", 32: "sports ball"}) == {0: s.PLAYER, 32: s.BALL}
    assert label_map(dict(enumerate(TARGET_CLASSES))) == {0: s.PLAYER, 1: s.REFEREE, 2: s.BALL}


def _write_split(root: Path, split: str, labels: dict[str, str]) -> None:
    (root / split / "images").mkdir(parents=True)
    (root / split / "labels").mkdir(parents=True)
    for stem, text in labels.items():
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (root / split / "labels" / f"{stem}.txt").write_text(text)


def test_remap_dataset_collapses_and_drops_classes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "data.yaml").write_text(
        yaml.safe_dump({"names": ["Ball", "Ref", "player-layup-dunk", "rim", "player"]})
    )
    _write_split(
        src,
        "train",
        {"a": "0 0.5 0.5 0.01 0.01\n1 0.2 0.2 0.1 0.3\n2 0.3 0.3 0.1 0.3\n3 0.9 0.1 0.1 0.1\n"},
    )
    _write_split(src, "valid", {"b": "4 0.5 0.5 0.1 0.3\n", "c": "3 0.9 0.1 0.1 0.1\n"})

    dst = tmp_path / "dst"
    counts = remap_dataset(src, dst)

    assert counts == {s.BALL: 1, s.REFEREE: 1, s.PLAYER: 2}
    assert (dst / "train/labels/a.txt").read_text().splitlines() == [
        "2 0.5 0.5 0.01 0.01",
        "1 0.2 0.2 0.1 0.3",
        "0 0.3 0.3 0.1 0.3",
    ]
    assert (dst / "valid/labels/c.txt").read_text() == ""  # rim dropped; kept as negative
    assert (dst / "valid/images/c.jpg").exists()
    data = yaml.safe_load((dst / "data.yaml").read_text())
    assert data["train"] == "train/images"
    assert data["val"] == "valid/images"
    assert data["names"] == dict(enumerate(TARGET_CLASSES))
