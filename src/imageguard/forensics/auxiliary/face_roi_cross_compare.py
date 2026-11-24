"""Cross-compare face ROI giữa khung nghi vấn và ảnh tham chiếu cùng phiên.

Ngữ cảnh (plan v2): operator lưu một khung webcam làm reference rồi phân tích
khung sau. Nếu vùng mặt ở khung sau mất vi kết cấu cảm biến so với reference
(da bị làm mịn kiểu AI/deepfake), cue tham chiếu tăng. Đây là tín hiệu khám
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

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import aux_working_copy
from imageguard.forensics.auxiliary.deepfake_cues import (
    _detect_face_roi,
    _fft_highfreq_cue,
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

# Trần điểm cộng thêm từ cue tham chiếu — cue chỉ được dịch chuyển nhẹ score
# gốc của DeepfakeCues, không át tín hiệu 6 trụ cột sẵn có.
REFERENCE_SCORE_MAX_DELTA = 0.15

# Chuẩn hoá từng đo lường về [0, 1] — span chọn theo cặp synthetic trong test.
_RESIDUAL_GAP_SPAN = 12.0  # độ lệch mức xám của mean-|residual| giữa hai ROI
_LBP_CHI2_BASE = 0.15  # nền χ² LBPH giữa hai lần chụp cùng một mặt
_LBP_CHI2_SPAN = 0.45
_FFT_PEAK_DELTA_SPAN = 4.0  # chênh đỉnh p99.8/median dải cao tần

_W_RESIDUAL = 0.30
_W_LBP = 0.45
_W_FFT = 0.25


def _gray_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)


def _face_roi_96(
    doc: ImageDocument, cfg: ThresholdSet
) -> tuple[np.ndarray, bool, str]:
    """ROI khuôn mặt của ``doc`` resize về 96×96 gray uint8.

    Dùng cùng working copy (aux_max_side) và cùng detector 3 tầng với
    ``DeepfakeCues`` nên vùng chọn nhất quán với kết quả chính.
    """
    work, _ = aux_working_copy(doc, cfg)
    arr = work.array
    gray = _gray_u8(arr)
    (x, y, rw, rh), detected, source = _detect_face_roi(arr, gray, _BLOCK)
    roi = gray[y : y + rh, x : x + rw]
    if roi.size == 0:  # rect thoát khỏi ảnh — fallback toàn khung
        roi = gray
    return (
        cv2.resize(roi, (ROI_SIDE, ROI_SIDE), interpolation=cv2.INTER_AREA),
        detected,
        source,
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
    reference: ImageDocument, suspect: ImageDocument
) -> dict:
    """So sánh ROI mặt 96×96 giữa ``reference`` và ``suspect``.

    Trả dict chỉ gồm hash + cờ + số đo (không đường dẫn): ``same_frame``,
    ``ref_combined_cue`` ∈ [0,1] và ba thành phần thô. Hai doc có array
    byte-identical → ``same_frame=True`` và ``ref_combined_cue == 0``.
    """
    cfg = load_thresholds()
    ref_roi, ref_detected, ref_source = _face_roi_96(reference, cfg)
    sus_roi, sus_detected, sus_source = _face_roi_96(suspect, cfg)

    same_frame = _same_array(reference, suspect)

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
        "same_frame": same_frame,
        "ref_combined_cue": round(combined, 4),
        "ref_residual_gap": round(residual_gap, 4),
        "ref_residual_cue": round(residual_cue, 4),
        "ref_lbp_chi2": round(lbp_chi2, 4),
        "ref_lbp_cue": round(lbp_cue, 4),
        "ref_fft_peak_delta": round(fft_delta, 4),
        "ref_fft_cue": round(fft_cue, 4),
        "ref_face_detected": bool(ref_detected),
        "sus_face_detected": bool(sus_detected),
        "ref_roi_source": ref_source,
        "sus_roi_source": sus_source,
        "reference_sha256": reference.sha256,
        "suspect_sha256": suspect.sha256,
    }


def apply_reference_cue(
    result: AlgorithmResult,
    reference: ImageDocument,
    suspect: ImageDocument,
) -> AlgorithmResult:
    """Áp cue tham chiếu lên kết quả deepfake có sẵn — trả AlgorithmResult mới.

    Luật (contract AUX): chỉ nâng ``NO_INDICATION`` → ``INCONCLUSIVE`` khi
    ``ref_combined_cue >= REFERENCE_CUE_MIN``; không bao giờ
    ``SUSPICIOUS_INDICATION``. Score tăng tối đa ``REFERENCE_SCORE_MAX_DELTA``
    và giữ nguyên khi hai khung trùng byte. Evidence chỉ gồm sha256 + số đo
    cue — không đường dẫn tuyệt đối. mask/heatmap/limitations/score_definition
    được sao chép nguyên từ ``result``.
    """
    t0 = time.perf_counter()
    cmp = compare_face_rois(reference, suspect)

    score = result.score
    verdict = result.verdict
    limitations = list(result.limitations)
    cue = float(cmp["ref_combined_cue"])

    if cmp["same_frame"]:
        limitations.append(
            "ảnh nghi vấn trùng byte với ảnh tham chiếu phiên — cue tham chiếu = 0"
        )
    else:
        if score is not None:
            score = round(
                min(1.0, score + REFERENCE_SCORE_MAX_DELTA * cue), 4
            )
        if verdict is Verdict.NO_INDICATION and cue >= REFERENCE_CUE_MIN:
            verdict = Verdict.INCONCLUSIVE
            limitations.append(
                f"vùng mặt lệch vi kết cấu so với ảnh tham chiếu phiên "
                f"(ref_cue={cue:.2f} ≥ {REFERENCE_CUE_MIN:.2f}, "
                f"residual_gap={cmp['ref_residual_gap']:.2f}, "
                f"lbp_χ²={cmp['ref_lbp_chi2']:.3f}) — cần đối chiếu thêm"
            )
        else:
            limitations.append(
                f"vùng mặt tương đồng ảnh tham chiếu phiên (ref_cue={cue:.2f})"
            )

    # Contract: evidence mới chỉ chứa sha256 + số đo cue — không path, không flag.
    evidence = dict(result.evidence)
    evidence.update(
        {
            "reference_sha256": cmp["reference_sha256"],
            "suspect_sha256": cmp["suspect_sha256"],
            "ref_combined_cue": cmp["ref_combined_cue"],
            "ref_residual_gap": cmp["ref_residual_gap"],
            "ref_lbp_chi2": cmp["ref_lbp_chi2"],
            "ref_fft_peak_delta": cmp["ref_fft_peak_delta"],
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
