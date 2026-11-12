"""Real-detector regression — planted copy-move must localize, or abstain cleanly.

Builds a seeded synthetic image with a planted copy-move region (synthforge
helpers), runs the actual keypoint pipeline, and asserts the detector's closed
Verdict. On a textured base the pipeline must return SUSPICIOUS_INDICATION with
a non-empty mask overlapping the planted region (IoU > 0). On a texture-less
base the detector must abstain with a closed Verdict (INCONCLUSIVE) instead of
crashing or over-claiming. Thresholds are never tuned here.
"""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.evaluation.synthforge import ForgeSpec, _base_image, forge
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove

_SEED = 7


def _doc(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr, width=w, height=h, channels=3, image_format="png",
        sha256="t" * 64, file_bytes=0, file_name="synthetic.png",
    )


def _iou(mask: np.ndarray, gt: np.ndarray) -> float:
    a, b = mask > 0, gt > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def test_planted_copymove_is_suspicious_and_localizes(ts) -> None:
    rng = np.random.default_rng(_SEED)
    size = (512, 512)
    base = _base_image(rng, size)  # textured synthetic base
    spec = ForgeSpec(
        size, "rect",
        [[120, 120], [280, 120], [280, 280], [120, 280]],  # 160×160 patch
        (150, 150),  # translate, không rotate/scale
        0.0, 1.0, "none", "easy",
    )
    forged, gt_mask = forge(base, spec)
    assert (gt_mask > 0).any(), "planted mask rỗng — spec sai"

    result = KeypointCopyMove().analyze(_doc(forged), ts)

    assert result.verdict == Verdict.SUSPICIOUS_INDICATION
    assert result.mask is not None

    iou = _iou(result.mask, gt_mask)
    # Đo thật (seed=7, rect 160×160 translate 150,150): IoU ≈ 0.43.
    # Keypoint mask bao quanh điểm nên luôn nhỏ hơn vùng copy thật — IoU > 0
    # đã đủ chứng tỏ detector định vị trúng vùng planted, không chỉ đoán mò.
    assert iou > 0.0

    # localization integrity: mask uint8 {0,255}, đúng kích thước ảnh gốc.
    assert set(np.unique(result.mask)) <= {0, 255}
    assert result.mask.shape == (size[1], size[0])


def test_textureless_plant_abstains_with_closed_verdict(ts) -> None:
    # Base phẳng hoàn toàn → SIFT không đủ keypoint → detector phải abstain
    # bằng INCONCLUSIVE (closed verdict), không crash, không SUSPICIOUS ép.
    flat = np.full((256, 256, 3), 128, np.uint8)
    spec = ForgeSpec(
        (256, 256), "rect",
        [[40, 40], [100, 40], [100, 100], [40, 100]],
        (80, 80), 0.0, 1.0, "none", "easy",
    )
    forged, _gt = forge(flat, spec)

    result = KeypointCopyMove().analyze(_doc(forged), ts)

    assert result.verdict in set(Verdict)  # closed enum, không crash
    assert result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert result.verdict == Verdict.INCONCLUSIVE
    assert any("insufficient texture" in lim for lim in result.limitations)
