"""Keypoint copy-move — regression trên synthforge tiers.

Acceptance nội bộ (ghi N mẫu đo thật, KHÔNG phải benchmark claim):
- easy tier: detector phải bắt được translate copy-move đa số mẫu.
- authentic: không được SUSPICIOUS ép.
- low-texture: INCONCLUSIVE + lý do.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.evaluation.synthforge import ForgeSpec, forge, generate_set
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr, width=w, height=h, channels=3, image_format="png",
        sha256="t" * 64, file_bytes=0, file_name="synthetic.png",
    )


def _iou(mask: np.ndarray, gt: np.ndarray) -> float:
    a, b = mask > 0, gt > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


@pytest.fixture(scope="module")
def synth_easy(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("easy")
    generate_set(5, seed=11, out_dir=d, tier="easy")
    return d


def test_easy_tier_detects_majority(synth_easy: Path, ts) -> None:
    method = KeypointCopyMove()
    hits = ious = 0
    n = 0
    for f in sorted(synth_easy.glob("*_F.png")):
        doc = safe_open(f, ts)
        assert isinstance(doc, ImageDocument)
        gt = cv2.imread(str(f).replace("_F.png", "_M.png"), 0)
        r = method.analyze(doc, ts)
        n += 1
        if r.verdict == Verdict.SUSPICIOUS_INDICATION and r.mask is not None:
            hits += 1
            ious += _iou(r.mask, gt)
    # ≥60% easy-tier → SUSPICIOUS và mask IoU trung bình > 0.15 (keypoint blob
    # bao quanh điểm, luôn nhỏ hơn vùng copy thật — IoU khiêm tốn là đúng).
    assert hits >= max(1, int(0.6 * n)), f"chỉ {hits}/{n} easy samples được bắt"
    assert ious / max(hits, 1) > 0.15


def test_authentic_no_false_suspicious(valid_png: Path, ts) -> None:
    doc = safe_open(valid_png, ts)
    assert isinstance(doc, ImageDocument)
    r = KeypointCopyMove().analyze(doc, ts)
    assert r.verdict != Verdict.SUSPICIOUS_INDICATION


def test_low_texture_inconclusive(ts) -> None:
    flat = np.full((256, 256, 3), 128, np.uint8)  # không texture → không keypoint
    r = KeypointCopyMove().analyze(_doc_from(flat), ts)
    assert r.verdict == Verdict.INCONCLUSIVE
    assert any("insufficient texture" in lim for lim in r.limitations)
    assert r.evidence["n_keypoints"] < int(ts.value("keypoint", "k_min"))


def test_rotate_scale_medium(ts) -> None:
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    spec = ForgeSpec((400, 400), "rect", [[50, 50], [150, 50], [150, 150], [50, 150]],
                     (140, 120), rotate_deg=15.0, scale=1.0, postproc="none", tier="medium")
    forged, mask = forge(base, spec)
    r = KeypointCopyMove().analyze(_doc_from(forged), ts)
    # Medium tier: chấp nhận SUSPICIOUS hoặc INCONCLUSIVE — không NO_INDICATION
    # (vùng copy 100×100 rotate 15° trên nền nhiễu ngẫu nhiên phải để lại dấu).
    assert r.verdict != Verdict.NO_INDICATION


def test_mask_maps_to_original_size(ts) -> None:
    base = np.random.default_rng(5).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    spec = ForgeSpec((300, 300), "rect", [[20, 20], [80, 20], [80, 80], [20, 80]],
                     (120, 100), 0.0, 1.0, "none", "easy")
    forged, _ = forge(base, spec)
    doc = _doc_from(forged)
    r = KeypointCopyMove().analyze(doc, ts)
    if r.mask is not None:
        assert r.mask.shape == (300, 300)
        assert set(np.unique(r.mask)) <= {0, 255}
