"""synthforge: deterministic + mask semantics đúng."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.synthforge import ForgeSpec, forge, generate_set


def test_deterministic_same_seed(tmp_path: Path) -> None:
    m1 = generate_set(3, 42, tmp_path / "a")
    m2 = generate_set(3, 42, tmp_path / "b")
    # manifest byte-identical — rng gieo cùng seed → spec giống hệt
    assert m1.read_bytes() == m2.read_bytes()


def test_forged_differs_and_mask_nonempty() -> None:
    base = np.random.default_rng(7).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    spec = ForgeSpec(
        size=(256, 256),
        shape="rect",
        src_polygon=[[10, 10], [60, 10], [60, 60], [10, 60]],
        dst_offset=(100, 80),
        rotate_deg=0.0,
        scale=1.0,
        postproc="none",
        tier="easy",
    )
    forged, mask = forge(base, spec)
    assert mask.sum() > 0
    assert not np.array_equal(forged, base)
    # Vùng dst trong forged phải giống vùng src (translate thuần)
    src = base[10:60, 10:60]
    dst = forged[90:140, 110:160]
    assert np.mean(np.abs(src.astype(int) - dst.astype(int))) < 1.0


def test_mask_is_binary(tmp_path: Path) -> None:
    generate_set(2, 1, tmp_path / "s")
    m = cv2.imread(str(tmp_path / "s" / "medium_0000_M.png"), 0)
    assert set(np.unique(m)) <= {0, 255}
