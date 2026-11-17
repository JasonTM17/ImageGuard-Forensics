"""Regression: Dual-Camera background isolation trong DeepfakeCues.

outer_bg / seam_ring phải luôn nằm trong cùng nửa camera đã chọn. Khi ROI chiếm
gần toàn bộ nửa đã chọn mà không còn nền cùng cảm biến, pipeline phải trả
INCONCLUSIVE (score/mask None) thay vì mượn nền camera còn lại.
"""

from __future__ import annotations

import numpy as np
import pytest

import imageguard.forensics.auxiliary.deepfake_cues as dc_mod
from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues

_HALF_H, _HALF_W = 240, 320
_FULL_HALF = (0, 0, _HALF_W, _HALF_H)
_MID_BX = 20  # bw = 640 // 16 = 40 block; mid_bx = 20


def _doc(arr: np.ndarray, name: str = "dual.png") -> ImageDocument:
    h, w = arr.shape[:2]
    return ImageDocument(
        array=arr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="d" * 64,
        file_bytes=int(arr.size),
        file_name=name,
    )


def _noise_half(h: int, w: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(120.0 + rng.normal(0.0, sigma, (h, w, 3)), 0, 255).astype(np.uint8)


def _dual_frame(left_sigma: float, right_sigma: float, seed: int = 42) -> np.ndarray:
    left = _noise_half(_HALF_H, _HALF_W, left_sigma, seed)
    right = _noise_half(_HALF_H, _HALF_W, right_sigma, seed + 1)
    return np.hstack([left, right])


class _SeqFaceROI:
    """_detect_face_roi giả lập: trả lần lượt (rect, face, src) theo thứ tự gọi trái→phải."""

    def __init__(self, returns: list[tuple[tuple[int, int, int, int], bool, str]]) -> None:
        self._returns = list(returns)
        self.calls = 0

    def __call__(self, work_arr, gray_u8, block_size):
        # Mỗi analyze() dual gọi đúng len(returns) lần → chỉ số xoay vòng theo cặp.
        r = self._returns[self.calls % len(self._returns)]
        self.calls += 1
        return r


def test_dual_full_half_roi_left_selected_never_borrows_right_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller repro: ROI ép full nửa trái; nhiễu camera phải không được rò vào σ_out."""
    monkeypatch.setattr(
        dc_mod,
        "_detect_face_roi",
        _SeqFaceROI([(_FULL_HALF, True, "controlled_full_face")]),
    )
    cfg = load_thresholds()
    for other_sigma in (2.0, 30.0):
        frame = _dual_frame(left_sigma=4.0, right_sigma=other_sigma)
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_camera_mode"] is True
        assert res.evidence["dual_roi_side"] == "left"
        assert res.evidence["roi_source"] == "dual_left:controlled_full_face"
        assert res.verdict is Verdict.INCONCLUSIVE
        assert res.score is None
        assert res.mask is None
        assert any("nền cùng cảm biến" in item for item in res.limitations)


def test_dual_full_half_roi_right_selected_never_borrows_left_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Đối xứng trái: chọn nửa phải full-ROI; nhiễu camera trái không được rò vào σ_out."""
    # Ép drop=0 để chọn nửa phải deterministic qua nhánh face_r & !face_l,
    # không phụ thuộc dao động median σ của vùng ROI trái nhỏ.
    real_stats = dc_mod._roi_sigma_stats

    def zero_drop(sigma_region, rect, bs_eff):
        med_in, med_bg, _ = real_stats(sigma_region, rect, bs_eff)
        return med_in, med_bg, 0.0

    monkeypatch.setattr(dc_mod, "_roi_sigma_stats", zero_drop)
    monkeypatch.setattr(
        dc_mod,
        "_detect_face_roi",
        _SeqFaceROI(
            [
                ((40, 40, 80, 80), False, "controlled_small"),
                (_FULL_HALF, True, "controlled_full_face"),
            ]
        ),
    )
    cfg = load_thresholds()
    for other_sigma in (2.0, 30.0):
        frame = _dual_frame(left_sigma=other_sigma, right_sigma=4.0)
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_camera_mode"] is True
        assert res.evidence["dual_roi_side"] == "right"
        assert res.evidence["roi_source"] == "dual_right:controlled_full_face"
        assert res.verdict is Verdict.INCONCLUSIVE
        assert res.score is None
        assert res.mask is None
        assert any("nền cùng cảm biến" in item for item in res.limitations)


def test_dual_near_full_roi_outer_bg_mask_stays_in_selected_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROI phủ gần hết nửa trái → cue arrays phải thuộc domain half-local đã chọn.

    Hợp đồng V2: ``_lbp_texture_cues`` nhận mảng của NỬA ĐÃ CHỌN (gray width =
    half pixel, mask block = lưới hoàn chỉnh của nửa đó) — mạnh hơn ràng buộc
    cũ "outer có cột = 0 ngoài nửa" vì miền ngoại cảm biến không còn tồn tại
    trong dữ liệu. RED trên V1 (gray 640, mask 40 cột toàn khung).
    """
    captured: dict[str, object] = {}
    real_lbp = dc_mod._lbp_texture_cues

    def spy(gray_u8, inside_roi, outer_bg, bs_eff, *a, **k):
        captured["gray_w"] = int(gray_u8.shape[1])
        captured["inside"] = inside_roi.copy()
        captured["outer"] = outer_bg.copy()
        return real_lbp(gray_u8, inside_roi, outer_bg, bs_eff)

    monkeypatch.setattr(dc_mod, "_lbp_texture_cues", spy)
    monkeypatch.setattr(
        dc_mod,
        "_detect_face_roi",
        _SeqFaceROI(
            [
                ((0, 0, 296, 240), True, "controlled_wide_face"),
                ((40, 40, 60, 60), False, "controlled_small"),
            ]
        ),
    )
    frame = _dual_frame(left_sigma=6.0, right_sigma=6.0)
    res = DeepfakeCues().analyze(_doc(frame), load_thresholds())

    assert res.evidence["dual_roi_side"] == "left"
    half_blocks = _HALF_W // 16
    assert captured["gray_w"] == _HALF_W, "pipeline cues phải chạy trên nửa đã chọn"
    assert "outer" in captured and bool(np.any(captured["outer"]))
    assert captured["outer"].shape[1] == half_blocks, (
        f"outer_bg {captured['outer'].shape[1]} cột ≠ {half_blocks} block nửa trái"
    )
    assert captured["inside"].shape == captured["outer"].shape
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}
    assert res.score is not None


def test_dual_valid_face_with_background_computes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: dual face vừa phải, còn nền cùng nửa → pipeline tính bình thường."""
    monkeypatch.setattr(
        dc_mod,
        "_detect_face_roi",
        _SeqFaceROI(
            [
                ((60, 50, 140, 120), True, "controlled_face"),
                ((40, 40, 60, 60), False, "controlled_small"),
            ]
        ),
    )
    frame = _dual_frame(left_sigma=8.0, right_sigma=8.0)
    res = DeepfakeCues().analyze(_doc(frame), load_thresholds())

    assert res.evidence["dual_camera_mode"] is True
    assert res.evidence["dual_roi_side"] == "left"
    assert res.score is not None
    assert "sigma_out_median" in res.evidence
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}


def test_single_image_fallback_behavior_unchanged() -> None:
    """Control: ảnh đơn (không dual) giữ nguyên fallback toàn cục cũ."""
    rng = np.random.default_rng(9)
    img = np.clip(128 + rng.normal(0.0, 16.0, (256, 256, 3)), 0, 255).astype(np.uint8)
    res = DeepfakeCues().analyze(_doc(img, "single.png"), load_thresholds())

    assert res.evidence["dual_camera_mode"] is False
    assert res.evidence["dual_roi_side"] is None
    assert res.verdict is Verdict.NO_INDICATION
    assert res.score is not None and res.score < 0.35
