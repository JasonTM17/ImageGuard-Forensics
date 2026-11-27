"""Cross-compare face ROI giữa khung nghi vấn và ảnh tham chiếu cùng phiên.

Ngữ cảnh (plan v2): operator lưu một khung webcam làm reference rồi phân tích
khung sau. Nếu vùng mặt ở khung sau mất vi kết cấu cảm biến so với reference
(da bị làm mịn kiểu AI/deepfake), cue tham chiếu tăng. Cue chỉ thay đổi
verdict khi cả hai ROI đều được phát hiện là mặt. Đây là tín hiệu khám
phá bổ sung cho ``DeepfakeCues`` — theo contract AUX_METHODS nó chỉ có thể
nâng ``NO_INDICATION`` → ``INCONCLUSIVE``, KHÔNG BAO GIỜ ``SUSPICIOUS``.

Ba đo lường trên ROI đã chuẩn hoá 96×96 (dùng lại ``_detect_face_roi`` và
``_fft_highfreq_cue`` của deepfake_cues, LBP/χ² của ``ipkit.features_basic``):

1. residual MAE — chênh lệch mức năng lượng nhiễu phần dư (|img − median3|)
   giữa hai ROI: mặt bị làm mịn mất nhiễu cảm biến → gap dương.
2. LBP χ² — khoảng cách chi-square giữa histogram LBPH (uniform 59-bin,
   lưới 4×4) của hai ROI.
3. FFT peak delta — chênh lệch đỉnh tuần hoàn cao tần (p99.8/median) của
   phổ 2D FFT giữa hai ROI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import aux_working_copy
from imageguard.forensics.auxiliary.deepfake_cues import (
    _clip_half_roi,
    _detect_face_roi,
    _fft_highfreq_cue,
    _setting,
)
from imageguard.ipkit import features_basic

ROI_SIDE = 96  # cạnh ROI chuẩn hoá — cố định miền so sánh độc lập kích thước ảnh
_BLOCK = 8  # block_size tối thiểu cho _detect_face_roi (min_dim = 2×8 = 32px)
_RESIDUAL_KERNEL = 3  # medianBlur k=3 — giữ nhiễu tần số cao trên ROI 96×96

# Ngưỡng cue tham chiếu: chỉ khi ref_combined_cue vượt mức này mới được nâng
# NO_INDICATION → INCONCLUSIVE. Con số được chứng minh trên cặp synthetic của
# tests/test_face_roi_cross_compare.py (khung mặt textured vs cùng mặt làm
# mịn Gaussian) — KHÔNG phải tỉ lệ chính xác trên dữ liệu thật.
REFERENCE_CUE_MIN = 0.50

# Giữ tên public để tương thích caller cũ. Đo tham chiếu riêng, không trộn
# vào điểm không gian của DeepfakeCues khi chưa hiệu chuẩn trên dữ liệu thật.
REFERENCE_SCORE_MAX_DELTA = 0.0

# Chuẩn hoá từng đo lường về [0, 1] — span chọn theo cặp synthetic trong test.
_RESIDUAL_GAP_SPAN = 12.0  # độ lệch mức xám của mean-|residual| giữa hai ROI
_LBP_CHI2_BASE = 0.15  # nền χ² LBPH giữa hai lần chụp cùng một mặt
_LBP_CHI2_SPAN = 0.45
_FFT_PEAK_DELTA_SPAN = 4.0  # chênh đỉnh p99.8/median dải cao tần

_W_RESIDUAL = 0.30
_W_LBP = 0.45
_W_FFT = 0.25


@dataclass(frozen=True, slots=True)
class _FaceSample:
    pixels: np.ndarray | None
    detected: bool
    source: str
    min_side: int
    contrast: float
    quality_reason: str


def _gray_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)


def _face_roi_96(
    doc: ImageDocument, cfg: ThresholdSet
) -> _FaceSample:
    """ROI khuôn mặt của ``doc`` resize về 96×96 gray uint8.

    Dùng cùng working copy (aux_max_side) và cùng detector 3 tầng với
    ``DeepfakeCues``. Kiểm kích thước và p95−p5 trên ROI làm việc trước khi
    resize: nội suy không được biến mặt nhỏ/thiếu tương phản thành dữ liệu tốt.
    """
    work, _ = aux_working_copy(doc, cfg)
    arr = work.array
    gray = _gray_u8(arr)
    rect, detected, source = _detect_face_roi(arr, gray, _BLOCK)
    x, y, rw, rh = _clip_half_roi(rect, gray.shape)
    roi = gray[y : y + rh, x : x + rw]
    min_side = min(rw, rh)
    contrast = float(np.percentile(roi, 95) - np.percentile(roi, 5))
    minimum_side = _setting(cfg, "min_face_side", 64, 16, 512)
    minimum_contrast = _setting(cfg, "min_roi_contrast", 8.0, 0, 255)
    reason = ""
    if not detected:
        reason = "no_face"
    elif min_side < minimum_side:
        reason = "roi_too_small"
    elif contrast < minimum_contrast:
        reason = "low_contrast"
    return _FaceSample(
        pixels=(
            cv2.resize(roi, (ROI_SIDE, ROI_SIDE), interpolation=cv2.INTER_AREA)
            if not reason else None
        ),
        detected=bool(detected),
        source=source,
        min_side=min_side,
        contrast=contrast,
        quality_reason=reason,
    )


def _residual_level(roi_u8: np.ndarray) -> float:
    """Mean-|residual| — mức năng lượng nhiễu phần dư trung bình của ROI."""
    f = roi_u8.astype(np.float32)
    residual = f - cv2.medianBlur(f, _RESIDUAL_KERNEL)
    return float(np.mean(np.abs(residual)))


def _fft_peak_ratio(roi_u8: np.ndarray) -> float:
    _, peak_ratio, _, _ = _fft_highfreq_cue(roi_u8.astype(np.float32))
    return peak_ratio


def _same_array(a: ImageDocument, b: ImageDocument) -> bool:
    arr_a, arr_b = a.array, b.array
    return (
        arr_a.shape == arr_b.shape
        and arr_a.dtype == arr_b.dtype
        and bool(np.array_equal(arr_a, arr_b))
    )


def compare_face_rois(
    reference: ImageDocument, suspect: ImageDocument, cfg: ThresholdSet | None = None
) -> dict:
    """So sánh ROI mặt 96×96 giữa ``reference`` và ``suspect``.

    Trả hash, cờ, lý do chất lượng và số đo (không đường dẫn). Chỉ đo cue khi
    cả hai ROI đạt kích thước/tương phản trên ảnh làm việc; nếu không, số đo
    cue là None. Hai ROI hợp lệ giống hệt → ``ref_combined_cue == 0``.
    """
    cfg = cfg or load_thresholds()
    ref = _face_roi_96(reference, cfg)
    sus = _face_roi_96(suspect, cfg)

    same_frame = _same_array(reference, suspect)
    comparison = {
        "same_frame": same_frame,
        "ref_combined_cue": None,
        "ref_residual_gap": None,
        "ref_residual_cue": None,
        "ref_lbp_chi2": None,
        "ref_lbp_cue": None,
        "ref_fft_peak_delta": None,
        "ref_fft_cue": None,
        "ref_face_detected": ref.detected,
        "sus_face_detected": sus.detected,
        "ref_roi_source": ref.source,
        "sus_roi_source": sus.source,
        "ref_quality_reason": ref.quality_reason,
        "sus_quality_reason": sus.quality_reason,
        "ref_roi_min_side": ref.min_side,
        "sus_roi_min_side": sus.min_side,
        "ref_roi_contrast": round(ref.contrast, 4),
        "sus_roi_contrast": round(sus.contrast, 4),
        "reference_sha256": reference.sha256,
        "suspect_sha256": suspect.sha256,
    }
    if ref.pixels is None or sus.pixels is None:
        return comparison
    ref_roi, sus_roi = ref.pixels, sus.pixels

    residual_gap = abs(_residual_level(ref_roi) - _residual_level(sus_roi))
    residual_cue = float(np.clip(residual_gap / _RESIDUAL_GAP_SPAN, 0.0, 1.0))

    lbp_chi2 = features_basic.chi_square_distance(
        features_basic.lbph(ref_roi, grid=(4, 4), uniform=True),
        features_basic.lbph(sus_roi, grid=(4, 4), uniform=True),
    )
    lbp_cue = float(
        np.clip((lbp_chi2 - _LBP_CHI2_BASE) / _LBP_CHI2_SPAN, 0.0, 1.0)
    )

    fft_delta = abs(_fft_peak_ratio(ref_roi) - _fft_peak_ratio(sus_roi))
    fft_cue = float(np.clip(fft_delta / _FFT_PEAK_DELTA_SPAN, 0.0, 1.0))

    combined = float(
        np.clip(
            _W_RESIDUAL * residual_cue + _W_LBP * lbp_cue + _W_FFT * fft_cue,
            0.0,
            1.0,
        )
    )
    if same_frame:
        # Trùng byte → mọi đo lường đã là 0; chốt rõ ràng để contract không
        # lệ thuộc dung sai số học.
        combined = 0.0

    return {
        **comparison,
        "ref_combined_cue": round(combined, 4),
        "ref_residual_gap": round(residual_gap, 4),
        "ref_residual_cue": round(residual_cue, 4),
        "ref_lbp_chi2": round(lbp_chi2, 4),
        "ref_lbp_cue": round(lbp_cue, 4),
        "ref_fft_peak_delta": round(fft_delta, 4),
        "ref_fft_cue": round(fft_cue, 4),
    }


def apply_reference_cue(
    result: AlgorithmResult,
    reference: ImageDocument,
    suspect: ImageDocument,
    cfg: ThresholdSet | None = None,
) -> AlgorithmResult:
    """Áp cue tham chiếu lên kết quả deepfake có sẵn — trả AlgorithmResult mới.

    Luật (contract AUX): chỉ nâng ``NO_INDICATION`` → ``INCONCLUSIVE`` khi
    ``ref_combined_cue >= REFERENCE_CUE_MIN``; không bao giờ
    ``SUSPICIOUS_INDICATION``. Score không gian giữ nguyên; cue tham chiếu có
    cờ và trạng thái riêng. Evidence không chứa đường dẫn tuyệt đối.
    mask/heatmap/limitations/score_definition
    được sao chép nguyên từ ``result``.
    """
    t0 = time.perf_counter()
    if (
        reference.capture_layout != "single"
        or suspect.capture_layout != "single"
        or result.evidence.get("dual_camera_mode")
    ):
        # Independent largest-face selection over a composite cannot guarantee
        # that reference and suspect belong to the same sensor or selected half.
        return AlgorithmResult(
            method=result.method,
            verdict=result.verdict,
            score=result.score,
            score_definition=result.score_definition,
            mask=result.mask,
            heatmap=result.heatmap,
            evidence={
                **result.evidence,
                "reference_cue_flag": False,
                "ref_status": "unsupported_capture_layout",
                "reference_sha256": reference.sha256,
                "suspect_sha256": suspect.sha256,
            },
            limitations=[
                *result.limitations,
                (
                    "cue tham chiếu chưa hỗ trợ ảnh ghép hai camera — cần chọn cùng "
                    "một cảm biến ở cả hai khung trước khi so sánh"
                ),
            ],
            params_hash=result.params_hash,
            duration_ms=result.duration_ms + int((time.perf_counter() - t0) * 1000),
        )
    cmp = compare_face_rois(reference, suspect, cfg)

    score = result.score
    verdict = result.verdict
    limitations = list(result.limitations)
    cue = cmp["ref_combined_cue"]

    both_faces = bool(cmp["ref_face_detected"] and cmp["sus_face_detected"])
    available = (
        not result.evidence.get("cue_unavailable")
        and not result.evidence.get("cancelled")
        and result.evidence.get("error") != "cancelled"
        and result.verdict not in {Verdict.ERROR, Verdict.NOT_APPLICABLE}
        and result.score is not None
    )
    quality_ok = not cmp["ref_quality_reason"] and not cmp["sus_quality_reason"]
    flagged = (
        available and both_faces and quality_ok and not cmp["same_frame"]
        and cue is not None and cue >= REFERENCE_CUE_MIN
    )
    if not available:
        status = "unavailable"
        limitations.append(
            "cue tham chiếu không bổ sung kết luận — phân tích vùng mặt gốc không khả dụng"
        )
    elif not both_faces:
        status = "face_unavailable"
        limitations.append(
            "cue tham chiếu không áp dụng — cần phát hiện mặt ở cả hai khung "
            f"(ref={int(bool(cmp['ref_face_detected']))}, "
            f"suspect={int(bool(cmp['sus_face_detected']))})"
        )
    elif not quality_ok:
        status = "quality_unavailable"
        for prefix, label in (("ref", "ảnh mốc"), ("sus", "ảnh đang xét")):
            reason = cmp[f"{prefix}_quality_reason"]
            if reason == "roi_too_small":
                limitations.append(
                    f"cue tham chiếu không áp dụng — vùng mặt {label} quá nhỏ "
                    f"({cmp[f'{prefix}_roi_min_side']}px trên ảnh làm việc trước resize)"
                )
            elif reason == "low_contrast":
                limitations.append(
                    f"cue tham chiếu không áp dụng — vùng mặt {label} thiếu tương phản "
                    f"(p95−p5={cmp[f'{prefix}_roi_contrast']:.2f} trước resize)"
                )
    elif cmp["same_frame"]:
        status = "same_frame"
        limitations.append(
            "ảnh nghi vấn trùng byte với ảnh tham chiếu phiên — cue tham chiếu = 0"
        )
    else:
        if flagged:
            status = "difference"
            if verdict is Verdict.NO_INDICATION:
                verdict = Verdict.INCONCLUSIVE
            limitations.append(
                f"vùng mặt lệch vi kết cấu so với ảnh tham chiếu phiên "
                f"(ref_cue={cue:.2f} ≥ {REFERENCE_CUE_MIN:.2f}, "
                f"residual_gap={cmp['ref_residual_gap']:.2f}, "
                f"lbp_χ²={cmp['ref_lbp_chi2']:.3f}) — cần đối chiếu thêm"
            )
        else:
            status = "no_difference_flag"
            limitations.append(
                f"chênh lệch vùng mặt dưới ngưỡng tham chiếu (ref_cue={cue:.2f}); "
                "không chứng minh cùng danh tính hay ảnh thật"
            )

    # Evidence độc lập với điểm không gian; không có đường dẫn nguồn.
    evidence = dict(result.evidence)
    evidence.update(
        {
            "reference_cue_flag": bool(flagged),
            "ref_status": status,
            "reference_sha256": cmp["reference_sha256"],
            "suspect_sha256": cmp["suspect_sha256"],
            "ref_combined_cue": cmp["ref_combined_cue"],
            "ref_residual_gap": cmp["ref_residual_gap"],
            "ref_lbp_chi2": cmp["ref_lbp_chi2"],
            "ref_fft_peak_delta": cmp["ref_fft_peak_delta"],
            "ref_face_detected": int(bool(cmp["ref_face_detected"])),
            "sus_face_detected": int(bool(cmp["sus_face_detected"])),
            **{
                key: cmp[key] for key in (
                    "ref_quality_reason", "sus_quality_reason", "ref_roi_min_side",
                    "sus_roi_min_side", "ref_roi_contrast", "sus_roi_contrast",
                )
            },
        }
    )

    return AlgorithmResult(
        method=result.method,
        verdict=verdict,
        score=score,
        score_definition=result.score_definition,
        mask=result.mask,
        heatmap=result.heatmap,
        evidence=evidence,
        limitations=limitations,
        params_hash=result.params_hash,
        duration_ms=result.duration_ms + int((time.perf_counter() - t0) * 1000),
    )
