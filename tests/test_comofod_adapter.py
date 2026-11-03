"""CoMoFoD adapter — discovery + absent-skip + dim verify với fixture giả."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.comofod import CoMoFoDAdapter


def _mk(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), np.uint8); mask[10:30, 10:30] = 255
    for sid in ("001", "002", "003"):
        cv2.imwrite(str(root / f"{sid}_F.png"), img)
        cv2.imwrite(str(root / f"{sid}_O.png"), img)
        cv2.imwrite(str(root / f"{sid}_B.png"), mask)


def test_absent_dataset_skips(tmp_path: Path) -> None:
    a = CoMoFoDAdapter(tmp_path / "khong_co")
    assert a.version() == "absent"
    assert list(a.samples("dev")) == []  # skip, không raise


def test_discovery_counts(tmp_path: Path) -> None:
    _mk(tmp_path / "comofod")
    a = CoMoFoDAdapter(tmp_path / "comofod")
    dev = list(a.samples("dev"))
    test = list(a.samples("test"))
    total = len(dev) + len(test)
    # mỗi id → forged + authentic (trừ khi split lệch)
    assert total == 6
    forged = [s for s in dev + test if s.label == "forged"]
    assert all(s.gt_mask_path is not None for s in forged)


def test_split_by_base_image(tmp_path: Path) -> None:
    _mk(tmp_path / "comofod")
    a = CoMoFoDAdapter(tmp_path / "comofod")
    ids_dev = {s.base_image_id for s in a.samples("dev")}
    ids_test = {s.base_image_id for s in a.samples("test")}
    assert ids_dev.isdisjoint(ids_test)  # không leakage: cùng id cùng phía


def test_verify_masks_dims(tmp_path: Path) -> None:
    _mk(tmp_path / "comofod")
    a = CoMoFoDAdapter(tmp_path / "comofod")
    rep = a.verify_masks()
    assert rep["checked"] >= 1 and not rep["dim_mismatch"]
