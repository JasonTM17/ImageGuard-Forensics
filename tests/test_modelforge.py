"""Tests cho modelforge — classic backend chạy luôn; LaMa skip nếu thiếu model."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.evaluation.modelforge import (
    KINDS,
    ClassicEditor,
    LaMaEditor,
    MODEL_PATH,
    forge,
)
from imageguard.evaluation.synthforge import _base_image


@pytest.fixture()
def base():
    rng = np.random.Generator(np.random.PCG64(3))
    return _base_image(rng, (256, 256))


@pytest.mark.parametrize("kind", KINDS)
def test_classic_forge_produces_gt_mask(base, kind):
    rng = np.random.Generator(np.random.PCG64(11))
    foreign = _base_image(rng, (256, 256))
    forged, mask, spec = forge(base, kind, ClassicEditor(), rng, foreign)
    assert forged.shape == base.shape
    assert mask.shape == base.shape[:2]
    gt_frac = float((mask > 0).mean())
    assert 0.005 < gt_frac < 0.35, f"{kind}: GT {gt_frac:.1%} bất thường"
    # Edit thật sự đổi pixel trong vùng GT
    diff = np.abs(forged.astype(int) - base.astype(int)).sum(axis=2)
    assert (diff[mask > 0] > 20).sum() > 50, f"{kind}: vùng GT không đổi"
    assert spec["kind"] == kind


def test_move_erases_source(base):
    """move: vùng gốc bị inpaint → pixel tại src khác base — twin bị phá."""
    rng = np.random.Generator(np.random.PCG64(5))
    forged, mask, spec = forge(base, "move", ClassicEditor(), rng)
    sx, sy, sw, sh = spec["src"]
    src_orig = base[sy:sy + sh, sx:sx + sw]
    src_new = forged[sy:sy + sh, sx:sx + sw]
    assert np.abs(src_orig.astype(int) - src_new.astype(int)).mean() > 5


def test_generate_set_contract(tmp_path):
    from imageguard.evaluation.modelforge import generate_set

    mf = generate_set(2, seed=1, out_dir=tmp_path / "mf",
                      size=(256, 256), backend="classic",
                      kinds=("erase", "splice_keep"))
    import json
    entries = json.loads(mf.read_text())
    assert len(entries) == 2
    for e in entries:
        stem = e["id"]
        ext = ".jpg" if e.get("format") == "jpeg" else ".png"
        for suf in ("_O.png", f"_F{ext}", "_M.png"):
            assert (mf.parent / f"{stem}{suf}").exists(), f"thiếu {stem}{suf}"


@pytest.mark.skipif(not MODEL_PATH.exists(),
                    reason="lama_fp32.onnx chưa tải (docs/modelforge.md)")
def test_lama_inpaint_fills_only_mask(base):
    rng = np.random.Generator(np.random.PCG64(7))
    mask = np.zeros((256, 256), np.uint8)
    cv2.ellipse(mask, (150, 150), (40, 30), 0, 0, 360, 255, -1)
    out = LaMaEditor().inpaint(base, mask)
    diff = np.abs(out.astype(int) - base.astype(int)).sum(axis=2)
    assert (diff[mask > 0] > 20).sum() > 100   # trong mask: nội dung mới
    assert diff[mask == 0].mean() < 5          # ngoài mask: gần như nguyên
