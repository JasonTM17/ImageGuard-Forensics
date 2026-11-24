"""Regression lock: extreme aspect ratios / 1px-wide images must never crash a
detector (cv2.error on 0-dim resize / medianBlur) nor emit a NaN score."""

from __future__ import annotations

import math

import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.freq_view import FreqView
from imageguard.forensics.auxiliary.noise_local import NoiseLocal
from imageguard.forensics.auxiliary.noise_map import NoiseMap
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove


def _rand_doc(h: int, w: int, seed: int) -> ImageDocument:
    array = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return ImageDocument(
        array=array,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="f" * 64,
        file_bytes=array.nbytes,
        file_name="extreme.png",
    )


def _assert_score_not_nan(score: float | None) -> None:
    # NOT_APPLICABLE kết quả có score=None là hợp lệ; cấm duy nhất là NaN.
    assert score is None or not math.isnan(score)


class _LargeMedianKProxy:
    """Cfg proxy: chỉ override noise.median_k=9 (cv2.medianBlur float32 chỉ
    nhận k lẻ 3/5 — giá trị 9 ở config cũ làm detector raise), còn lại
    delegate nguyên ThresholdSet bên dưới."""

    def __init__(self, base: ThresholdSet) -> None:
        self._base = base

    def value(self, section: str, key: str):
        if (section, key) == ("noise", "median_k"):
            return 9
        return self._base.value(section, key)

    def limit(self, key: str, fallback: int) -> int:
        return self._base.limit(key, fallback)

    def snapshot_hash(self) -> str:
        return self._base.snapshot_hash()


def test_block_detector_survives_extreme_aspect_ratio(ts) -> None:
    # 3×4000: scale 768/4000 → cạnh ngắn int(3*scale)=0 ở code cũ → cv2.error.
    doc = _rand_doc(3, 4000, seed=11)
    result = BlockCopyMove().analyze(doc, ts)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.mask is None or result.mask.shape == (3, 4000)
    assert result.heatmap is None or result.heatmap.shape == (3, 4000)


def test_keypoint_detector_survives_extreme_aspect_ratio(ts) -> None:
    # 1×4000: scale 2048/4000 → int(1*scale)=0 ở code cũ → cv2.error.
    doc = _rand_doc(1, 4000, seed=12)
    result = KeypointCopyMove().analyze(doc, ts)
    assert isinstance(result.verdict, Verdict)
    assert result.mask is None or result.mask.shape == (1, 4000)


def test_freq_view_one_pixel_wide_image_has_no_nan_score(ts) -> None:
    # 50×1: np.diff theo trục ngang rỗng → mean NaN lọt vào score ở code cũ.
    doc = _rand_doc(50, 1, seed=13)
    result = FreqView().analyze(doc, ts)
    assert result.score is not None
    assert not math.isnan(result.score)
    assert result.evidence["blockiness"] == 0.0
    for value in result.evidence.values():
        if isinstance(value, float):
            assert not math.isnan(value)


def test_noise_detectors_clamp_large_median_k_cfg(ts) -> None:
    doc = _rand_doc(64, 64, seed=14)
    proxy = _LargeMedianKProxy(ts)

    for detector in (NoiseMap(), NoiseLocal()):
        result = detector.analyze(doc, proxy)  # median_k=9 phải bị clamp về 5
        _assert_score_not_nan(result.score)

    # Sanity: với config shipped (median_k=5) cả hai vẫn chạy bình thường.
    for detector in (NoiseMap(), NoiseLocal()):
        result = detector.analyze(doc, ts)
        _assert_score_not_nan(result.score)
