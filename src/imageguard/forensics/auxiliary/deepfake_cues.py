"""Deepfake / AI-manipulation cues (tín hiệu khám phá, KHÔNG phải bộ phân loại DL).

Pipeline 6 trụ cột Xử lý ảnh số (thuần NumPy + OpenCV + ``imageguard.ipkit``):
  1. Định vị vùng khuôn mặt (ROI): Haar Cascade nếu có, hoặc phân đoạn màu da
     YCrCb + HSV + hình thái học + kiểm chứng cấu trúc mắt/đối xứng SAT.
  2. Chênh lệch nhiễu phần dư cảm biến σ_in vs σ_out và bất liên tục vòng
     biên ghép seam-ring, kết hợp phân ngưỡng Otsu & thành phần liên thông.
  3. Vi cấu trúc texture bằng Local Binary Pattern (LBP) 8 lân cận và khoảng
     cách histogram χ² giữa ROI với nền.
  4. Dấu vết phổ tần số 2D FFT: đỉnh tuần hoàn lưới upsampling & tỷ lệ suy hao
     phổ hướng tâm dải cao tần.
  5. Biên ghép hình học quanh viền ROI: Canny Edges + trích biên hình thái
     học (boundary extraction) + biến đổi Hough tìm đoạn biên thẳng bất thường.
  6. Đối xứng cấu trúc khuôn mặt trái–phải bằng đặc trưng HOG & 7 bất biến Hu.

Hỗ trợ ảnh ghép hai camera khi có capture_layout rõ ràng, không đoán từ tỷ lệ ảnh.

Contract bắt buộc: thuộc ``AUX_METHODS`` — chỉ trả ``INCONCLUSIVE``,
``NO_INDICATION`` hoặc ``NOT_APPLICABLE``; KHÔNG BAO GIỜ trả ``SUSPICIOUS_INDICATION``.
"""

from __future__ import annotations

import threading
import time
from numbers import Real
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.core.preprocess import (
    aux_limitation,
    aux_working_copy,
    scaled_block,
    scaled_kernel,
    upscale_to_doc,
)
from imageguard.forensics.base import cancelled
from imageguard.ipkit import (
    features_basic,
    features_hog,
    morph_algos,
    morphology,
    segmentation,
    threshold,
)

SCORE_DEF = (
    "chỉ số kết hợp 6 trụ cột Xử lý ảnh (ROI da+SAT, nhiễu dư σ-Otsu, "
    "LBP χ², phổ 2D FFT, Canny-Hough biên ghép & đối xứng HOG/Hu) — "
    "chưa hiệu chuẩn, không phải xác suất Deepfake/AI"
)


def _setting(
    cfg: ThresholdSet, key: str, default: float, low: float = 0.0, high: float = 1.0,
) -> float:
    """Read the normal threshold schema and validate legacy scalar overrides.

    Earlier callers supplied scalars in ``cfg.raw['deepfake']``. Retain that
    API, but never allow NaN/bools/out-of-range values to silently turn flags off.
    """
    section = cfg.raw.get("deepfake", {})
    if not isinstance(section, dict):
        raise ValueError("deepfake phải là mapping ngưỡng")  # noqa: TRY004 - matches ThresholdSet
    node = section.get(key, default)
    value = cfg.value("deepfake", key) if isinstance(node, dict) else node
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"deepfake.{key} phải là số hữu hạn")  # noqa: TRY004 - matches ThresholdSet
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"deepfake.{key} phải nằm trong [{low}, {high}]")
    return value


def _capture_split(doc: ImageDocument) -> int | None:
    """Return an explicitly declared camera boundary in document pixels."""
    if doc.capture_layout == "single":
        return None
    if doc.capture_layout != "dual_side_by_side":
        raise ValueError("capture_layout không được hỗ trợ")
    split = doc.capture_split_x if doc.capture_split_x is not None else doc.width // 2
    if isinstance(split, bool) or not isinstance(split, int) or not 0 < split < doc.width:
        raise ValueError("capture_split_x phải nằm bên trong ảnh ghép")
    return split


def _load_haar_cascade():
    """Return a usable detector and explicit status for custom OpenCV builds."""
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if not callable(cascade_cls):
        return None, "missing_classifier"
    haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    if not haar_dir:
        return None, "missing_cascade_data"
    cascade_path = Path(haar_dir) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return None, "missing_cascade_data"
    try:
        cascade = cascade_cls(str(cascade_path))
        if cascade.empty():
            return None, "cascade_load_failed"
    except (cv2.error, OSError, ValueError):
        return None, "cascade_load_failed"
    return cascade, "available"

# 8 láng giềng ngược/cùng chiều kim đồng hồ khớp quy tắc ipkit/features_basic.py.
_LBP_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
)


def _block_means_from_integral(integ: np.ndarray, bh: int, bw: int, bs: int) -> np.ndarray:
    """Tính giá trị trung bình từng block bs x bs từ ảnh tích phân ``cv2.integral``."""
    ys, xs = np.mgrid[0:bh, 0:bw] * bs
    sums = (
        integ[ys + bs, xs + bs]
        - integ[ys, xs + bs]
        - integ[ys + bs, xs]
        + integ[ys, xs]
    )
    return (sums / float(bs * bs)).astype(np.float32)


def _verify_face_candidate_sat(gray_u8: np.ndarray, x: int, y: int, rw: int, rh: int) -> bool:
    """Require two localized eye-like regions, not skin color or symmetry alone.

    This conservative classical fallback is for roughly frontal faces. It can
    miss profiles, occluded eyes and strong pose changes; no match is preferable
    to labeling a shaded wall/arm/ball as a face. These geometric checks are
    exploratory defaults, not a learned face detector or a real-world accuracy
    claim. Haar cascade candidates do not depend on skin-color thresholds.
    """
    roi = gray_u8[y : y + rh, x : x + rw]
    if roi.shape[0] < 24 or roi.shape[1] < 24:
        return False

    norm = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    norm = cv2.GaussianBlur(norm, (3, 3), 0.8)
    integ = cv2.integral(norm)

    def _box_mean(x0: int, y0: int, x1: int, y1: int) -> float:
        area = float(max(1, (x1 - x0) * (y1 - y0)))
        s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        return float(s / area)

    left = norm[:, :32]
    right = norm[:, 32:][:, ::-1]
    left_c = left - float(np.mean(left))
    right_c = right - float(np.mean(right))
    l1_sym = 1.0 - float(np.mean(np.abs(left_c - right_c))) / 128.0
    if l1_sym < 0.55:
        return False

    # A broad intensity gradient has no localized dark response. A horizontal
    # stripe remains one connected region, unlike two separated eyes. Compute
    # components before assigning left/right so a stripe cannot masquerade as
    # one eye in each independently cropped window.
    contrast = float(np.percentile(norm, 95) - np.percentile(norm, 5))
    min_delta = max(3.0, 0.06 * contrast)
    local_mean = cv2.GaussianBlur(norm, (0, 0), 3.0)
    dark = ((local_mean - norm) >= min_delta).astype(np.uint8)
    dark[:10, :] = 0
    dark[38:, :] = 0
    dark[:, :4] = 0
    dark[:, 60:] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
    eyes = []
    for index in range(1, count):
        ex, ey, ew, eh, area = (int(value) for value in stats[index])
        cx, cy = (float(value) for value in centers[index])
        if not (6 <= area <= 160 and 3 <= ew <= 22 and 2 <= eh <= 15):
            continue
        if ey <= 10 or ey + eh >= 38:
            continue
        if not (10 <= cx <= 27 or 37 <= cx <= 54):
            continue
        eyes.append((cx, cy, area, _box_mean(ex, ey, ex + ew, ey + eh)))

    for lx, ly, la, left_eye in eyes:
        if lx > 27:
            continue
        for rx_eye, ry_eye, ra, right_eye in eyes:
            if rx_eye < 37 or abs(ly - ry_eye) > 7:
                continue
            if not (16 <= rx_eye - lx <= 42) or min(la, ra) / max(la, ra) < 0.25:
                continue
            mx, my = round((lx + rx_eye) / 2), round((ly + ry_eye) / 2)
            bridge = _box_mean(mx - 3, my - 3, mx + 4, my + 4)
            cheek_l = _box_mean(10, 36, 28, 49)
            cheek_r = _box_mean(36, 36, 54, 49)
            if (
                cheek_l - left_eye >= min_delta
                and cheek_r - right_eye >= min_delta
                and bridge - max(left_eye, right_eye) >= min_delta
            ):
                return True

    return False


def _detect_face_roi(
    work_arr: np.ndarray,
    gray_u8: np.ndarray,
    block_size: int,
    cfg: ThresholdSet | None = None,
) -> tuple[tuple[int, int, int, int], bool, str]:
    """Prefer measurable face candidates, then a conservative skin/SAT fallback.

    ``cfg`` is optional for older callers. Candidate ranking uses the same
    quality thresholds as the downstream analysis, including user overrides.
    """
    h, w = gray_u8.shape
    min_dim = max(32, block_size * 2)
    cfg = cfg or load_thresholds()
    min_side = _setting(cfg, "min_face_side", 64, 16, 512)
    min_contrast = _setting(cfg, "min_roi_contrast", 8.0, 0, 255)
    min_reference = _setting(cfg, "min_reference_blocks", 4, 1, 4096)

    def candidate_key(rect):
        x, y, rw, rh = rect
        crop = gray_u8[y:y + rh, x:x + rw]
        contrast = float(np.percentile(crop, 95) - np.percentile(crop, 5))
        bh, bw = h // block_size, w // block_size
        reference_count = 0
        if bh and bw:
            _, _, reference = _roi_masks((bh, bw), rect, block_size)
            reference_count = int(np.count_nonzero(reference))
        checks = (
            min(rw, rh) >= min_side,
            contrast >= min_contrast,
            reference_count >= min_reference,
        )
        return all(checks), sum(checks), rw * rh, contrast, -y, -x

    # Tầng 1: OpenCV Haar Cascade (nếu bản build hỗ trợ và tìm được file XML).
    haar_rect = None
    cascade, _status = _load_haar_cascade()
    if cascade is not None:
        faces = cascade.detectMultiScale(
            gray_u8,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_dim, min_dim),
        )
        if len(faces) > 0:
            candidates = [
                _clip_half_roi(tuple(int(value) for value in rect), (h, w))
                for rect in faces
            ]
            haar_rect = max(candidates, key=candidate_key)
            if candidate_key(haar_rect)[0]:
                return haar_rect, True, "haar_cascade"

    # Tầng 2: Phân đoạn màu da YCrCb + HSV + hình thái học (C4) + kiểm chứng Haar SAT (C6).
    if work_arr.ndim == 3 and work_arr.shape[2] >= 3:
        bgr = work_arr[:, :, :3]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask_ycrcb = (
            (ycrcb[:, :, 0] >= 40)
            & (ycrcb[:, :, 1] >= 133)
            & (ycrcb[:, :, 1] <= 178)
            & (ycrcb[:, :, 2] >= 77)
            & (ycrcb[:, :, 2] <= 128)
        )
        mask_hsv = (
            ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] >= 25)
            & (hsv[:, :, 1] <= 210)
            & (hsv[:, :, 2] >= 45)
        )
        skin_mask = (mask_ycrcb & mask_hsv).astype(np.uint8) * 255

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k_open)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k_close)

        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(skin_mask, 8)
        min_area = max(min_dim * min_dim, int(0.015 * h * w))
        best_rect: tuple[int, int, int, int] | None = None
        for i in range(1, n_lbl):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            rw = int(stats[i, cv2.CC_STAT_WIDTH])
            rh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or rw < min_dim or rh < min_dim:
                continue
            aspect = rw / float(max(rh, 1))
            if not (0.45 <= aspect <= 1.85):
                continue
            if _verify_face_candidate_sat(gray_u8, x, y, rw, rh):
                candidate = (x, y, rw, rh)
                if best_rect is None or candidate_key(candidate) > candidate_key(best_rect):
                    best_rect = candidate

        if best_rect is not None and (haar_rect is None or candidate_key(best_rect)[0]):
            return best_rect, True, "skin_haar_dip"

    if haar_rect is not None:
        # Preserve a genuine cascade detection when no candidate is usable;
        # the analysis reports its exact quality failure instead of a score.
        return haar_rect, True, "haar_cascade"

    # Tầng 3: Fallback về vùng trung tâm khi ảnh xám hoặc không có khuôn mặt.
    return (w // 4, h // 4, max(1, w // 2), max(1, h // 2)), False, "central_fallback"


def _lbp_texture_cues(
    gray_u8: np.ndarray,
    inside_roi: np.ndarray,
    outer_bg: np.ndarray,
    bs_eff: int,
) -> tuple[float, float, float, float, np.ndarray]:
    """Tính vi kết cấu LBP 8 lân cận vector hóa, độ phẳng da & khoảng cách χ² (C6)."""
    h, w = gray_u8.shape
    bh, bw = inside_roi.shape
    padded = np.pad(gray_u8, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    lbp_codes = np.zeros((h, w), dtype=np.uint8)
    for bit_idx, (dx, dy) in enumerate(_LBP_OFFSETS):
        nbr = padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        lbp_codes |= ((nbr >= center).astype(np.uint8) << np.uint8(bit_idx))

    # Đo biên độ vi mô 3x3: vùng bị làm mịn AI / Gaussian mất lỗ chân lông sẽ phẳng.
    k3 = np.ones((3, 3), dtype=np.uint8)
    micro_range = cv2.subtract(cv2.dilate(gray_u8, k3), cv2.erode(gray_u8, k3))
    flat_px = ((lbp_codes == 0) | (lbp_codes == 255) | (micro_range <= 2)).astype(np.float32)
    flat_blocks = _block_means_from_integral(cv2.integral(flat_px), bh, bw, bs_eff)

    flat_in = float(np.median(flat_blocks[inside_roi])) if np.any(inside_roi) else 0.0
    flat_out = float(np.median(flat_blocks[outer_bg])) if np.any(outer_bg) else flat_in

    # Mở rộng mask block thành mask pixel để dựng histogram LBP 16-bin cho ROI vs nền ngoài.
    in_px = np.repeat(np.repeat(inside_roi, bs_eff, axis=0), bs_eff, axis=1)
    out_px = np.repeat(np.repeat(outer_bg, bs_eff, axis=0), bs_eff, axis=1)
    crop_h, crop_w = in_px.shape
    bins_img = (lbp_codes[:crop_h, :crop_w] >> 4).ravel()

    in_vals = bins_img[in_px.ravel()]
    out_vals = bins_img[out_px.ravel()]
    if in_vals.size > 0 and out_vals.size > 0:
        h_in = np.bincount(in_vals, minlength=16).astype(np.float64)
        h_out = np.bincount(out_vals, minlength=16).astype(np.float64)
        h_in /= max(float(h_in.sum()), 1.0)
        h_out /= max(float(h_out.sum()), 1.0)
        lbp_chi2 = float(0.5 * np.sum(((h_in - h_out) ** 2) / (h_in + h_out + 1e-6)))
    else:
        lbp_chi2 = 0.0

    # Chỉ báo LBP tăng khi trong ROI mất vi kết cấu hơn nền VÀ/HOẶC phân phối LBP lệch nền.
    flat_delta = max(0.0, flat_in - flat_out)
    flat_cue = float(np.clip(flat_delta / max(1.0 - flat_out, 0.25) * 1.4, 0.0, 1.0))
    chi2_cue = float(np.clip((lbp_chi2 - 0.04) / 0.26, 0.0, 1.0))
    lbp_cue = float(np.clip(0.55 * flat_cue + 0.45 * chi2_cue, 0.0, 1.0))
    return lbp_cue, lbp_chi2, flat_in, flat_out, flat_blocks


def _fft_highfreq_cue(roi_gray: np.ndarray) -> tuple[float, float, float, float]:
    """Measure spectral anomalies; these are not Deepfake probabilities.

    Return ``(freq_cue, peak_ratio, spectral_rolloff, hf_conc8)``. The last
    value remains the legacy top-eight *amplitude* share for diagnostics;
    it depends on ROI size and is not used to make a decision.

    Periodicity uses the power in a fixed top 1% of high-frequency bins.
    For ideal white noise, each periodogram power is exponentially
    distributed, so the mean of its upper fraction q relative to the whole
    mean is ``1 - log(q)``. Normalize by that reference before scoring.
    A Hann window makes neighboring bins dependent; the twofold margin
    below is a conservative exploratory heuristic, not a significance test
    or a threshold calibrated on real/fake photographs.
    """
    rh, rw = roi_gray.shape
    if rh < 16 or rw < 16 or np.ptp(roi_gray) == 0:
        # Flat crops have no AC spectrum from which to infer a rolloff.
        return 0.0, 1.0, 1.0, 0.0
    centered = roi_gray.astype(np.float32) - float(roi_gray.mean())
    win = np.outer(np.hanning(rh), np.hanning(rw)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(centered * win)))
    spec = np.log1p(mag)

    # fftshift places DC at integer floor(size / 2), including even crops.
    yy, xx = np.ogrid[:rh, :rw]
    cy, cx = rh // 2, rw // 2
    norm_r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    hf_mask = (norm_r >= 0.45) & (norm_r <= 0.95)
    mf_mask = (norm_r >= 0.15) & (norm_r < 0.45)
    if not np.any(hf_mask):
        return 0.0, 1.0, 1.0, 0.0

    hf_mag = mag[hf_mask]
    hf_med = float(np.median(hf_mag)) + 1e-6
    hf_p998 = float(np.percentile(hf_mag, 99.8))
    peak_ratio = hf_p998 / hf_med

    # Retain this raw statistic for existing report readers only. A fixed
    # count of bins captures a larger share of noise in a smaller crop.
    raw_top_count = min(8, hf_mag.size)
    raw_top = np.partition(hf_mag, hf_mag.size - raw_top_count)[-raw_top_count:]
    hf_conc8 = float(raw_top.sum() / (hf_mag.sum() + 1e-9))

    # Actual spectral energy is |FFT|**2, not |FFT|. A fixed tail fraction
    # and noise reference make the periodicity statistic comparable across
    # crop areas without resizing away or inventing frequency artifacts.
    TAIL_FRACTION = 0.01
    NOISE_ENRICHMENT_FLOOR = 2.0
    ENRICHMENT_FULL_SCALE = 6.0
    hf_power = np.square(hf_mag.astype(np.float64))
    tail_count = max(1, int(np.ceil(TAIL_FRACTION * hf_power.size)))
    tail = np.partition(hf_power, hf_power.size - tail_count)[-tail_count:]
    actual_fraction = tail_count / hf_power.size
    tail_share = float(tail.sum() / max(float(hf_power.sum()), 1e-12))
    expected_noise_enrichment = 1.0 - np.log(actual_fraction)
    enrichment = (tail_share / actual_fraction) / expected_noise_enrichment
    grid_score = float(np.clip(
        (enrichment - NOISE_ENRICHMENT_FLOOR)
        / (ENRICHMENT_FULL_SCALE - NOISE_ENRICHMENT_FLOOR),
        0.0, 1.0,
    ))

    hf_mean = float(np.mean(spec[hf_mask]))
    mf_mean = float(np.mean(spec[mf_mask])) if np.any(mf_mask) else hf_mean
    spectral_rolloff = hf_mean / (mf_mean + 1e-6)
    peak_score = float(np.clip((peak_ratio - 4.0) / 14.0, 0.0, 1.0))
    rolloff_score = float(np.clip((0.58 - spectral_rolloff) / 0.25, 0.0, 1.0))
    freq_cue = float(
        np.clip(
            # A narrow periodic grid can occupy fewer than 0.2% of bins,
            # especially in large crops. Its power evidence must not be
            # capped at 0.35 merely because the percentile peak misses it.
            max(peak_score, grid_score, 0.65 * peak_score + 0.35 * rolloff_score),
            0.0,
            1.0,
        )
    )
    return freq_cue, peak_ratio, spectral_rolloff, hf_conc8


def _seam_canny_hough_cue(roi_gray_u8: np.ndarray) -> tuple[float, int]:
    """Trụ cột 5: Dò đường biên ghép thẳng quanh viền ROI (Canny + Boundary + Hough)."""
    rh, rw = roi_gray_u8.shape[:2]
    if rh < 24 or rw < 24:
        return 0.0, 0
    scale = min(1.0, 96.0 / float(max(rh, rw)))
    hs, ws = max(24, round(rh * scale)), max(24, round(rw * scale))
    roi_s = cv2.resize(roi_gray_u8, (ws, hs), interpolation=cv2.INTER_AREA)

    edge_canny = segmentation.canny_edges(roi_s, low=20.0, high=60.0, sigma=1.2)
    edge_bin = np.where(edge_canny > 0, 255, 0).astype(np.uint8)
    se = morphology.structuring_element("cross", 3)
    thin_edges = morph_algos.boundary_extraction(edge_bin, se)

    # Chỉ giữ vành 22% sát viền hộp ROI để bắt đường cắt/khung ghép mặt nạ.
    border_mask = np.ones((hs, ws), dtype=bool)
    mh = max(3, int(hs * 0.22))
    mw = max(3, int(ws * 0.22))
    if 2 * mh < hs and 2 * mw < ws:
        border_mask[mh : hs - mh, mw : ws - mw] = False
    border_edges = (np.asarray(thin_edges) > 0) & border_mask
    if int(np.count_nonzero(border_edges)) < 8:
        return 0.0, 0

    _rhos, thetas, votes = segmentation.hough_lines(border_edges, theta_step_deg=3.0, rho_step=2.0)
    if len(votes) == 0:
        return 0.0, 0

    min_votes = max(10, int(0.28 * min(hs, ws)))
    axis_lines = 0
    max_ratio = 0.0
    for theta_rad, v in zip(thetas[:12], votes[:12]):
        if int(v) < min_votes:
            continue
        deg = float(np.rad2deg(theta_rad)) % 180.0
        near_h = min(deg, 180.0 - deg) <= 12.0
        near_v = abs(deg - 90.0) <= 12.0
        if near_h or near_v:
            axis_lines += 1
            max_ratio = max(max_ratio, float(v) / float(max(min(hs, ws), 1)))
    if axis_lines == 0:
        return 0.0, 0
    cue = float(np.clip(0.55 * min(1.0, axis_lines / 4.0) + 0.45 * min(1.0, (max_ratio - 0.28) / 0.45), 0.0, 1.0))
    return cue, axis_lines


def _facial_hog_symmetry_cue(roi_gray_u8: np.ndarray) -> tuple[float, float]:
    """Trụ cột 6: Đo bất đối xứng cấu trúc trái–phải ROI bằng HOG & 7 bất biến Hu."""
    rh, rw = roi_gray_u8.shape[:2]
    if rh < 32 or rw < 32:
        return 0.0, 0.0
    r64 = cv2.resize(roi_gray_u8, (64, 64), interpolation=cv2.INTER_AREA)
    left = r64[:, :32]
    right_flip = np.ascontiguousarray(np.fliplr(r64[:, 32:]))

    hog_l = features_hog.hog_descriptor(left, cell=8, block=2, bins=9)
    hog_r = features_hog.hog_descriptor(right_flip, cell=8, block=2, bins=9)
    denom = float(np.linalg.norm(hog_l) + np.linalg.norm(hog_r)) + 1e-6
    hog_asym = float(np.linalg.norm(hog_l - hog_r) / denom)

    if int(np.sum(left)) == 0 or int(np.sum(right_flip)) == 0:
        hu_asym = 0.0 if int(np.sum(left)) == int(np.sum(right_flip)) else 1.0
    else:
        hu_l = np.log10(np.abs(features_basic.hu_moments(left)) + 1e-12)
        hu_r = np.log10(np.abs(features_basic.hu_moments(right_flip)) + 1e-12)
        hu_asym = float(np.mean(np.abs(hu_l - hu_r)) / (np.mean(np.abs(hu_l)) + 1.0))

    sym_cue = float(np.clip((0.70 * hog_asym + 0.30 * hu_asym - 0.28) / 0.42, 0.0, 1.0))
    return sym_cue, hog_asym


def _roi_sigma_stats(
    sigma_region: np.ndarray, rect: tuple[int, int, int, int], bs_eff: int
) -> tuple[float, float, float]:
    """Tính σ trung vị trong ROI, ngoài ROI và độ suy giảm mịn cục bộ trong một vùng."""
    bh_r, bw_r = sigma_region.shape
    rx, ry, rw, rh = rect
    bx0 = min(bw_r - 1, max(0, rx // bs_eff))
    by0 = min(bh_r - 1, max(0, ry // bs_eff))
    bx1 = min(bw_r, max(bx0 + 1, (rx + rw + bs_eff - 1) // bs_eff))
    by1 = min(bh_r, max(by0 + 1, (ry + rh + bs_eff - 1) // bs_eff))
    roi_m = np.zeros((bh_r, bw_r), dtype=bool)
    roi_m[by0:by1, bx0:bx1] = True
    bg_m = ~roi_m
    if not np.any(bg_m):
        bg_m = roi_m
    med_in = float(np.median(sigma_region[roi_m]))
    med_bg = float(np.median(sigma_region[bg_m]))
    drop = max(0.0, (med_bg - med_in) / (med_bg + 1e-6)) if med_bg > 1.0 else 0.0
    return med_in, med_bg, drop


def _sigma_grid(gray: np.ndarray, kernel: int, block: int) -> np.ndarray:
    """Residual/block statistics on one sensor's own pixel domain."""
    residual = gray - cv2.medianBlur(gray, kernel).astype(np.float32)
    bh, bw = gray.shape[0] // block, gray.shape[1] // block
    e2 = _block_means_from_integral(cv2.integral(residual * residual), bh, bw, block)
    e1 = _block_means_from_integral(cv2.integral(residual), bh, bw, block)
    return np.sqrt(np.clip(e2 - e1 * e1, 0.0, None)).astype(np.float32)


def _roi_masks(
    shape: tuple[int, int], rect: tuple[int, int, int, int], block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint ROI, adjacent ring and reference blocks on one sensor only."""
    bh, bw = shape
    rx, ry, rw, rh = rect
    bx0, by0 = min(bw - 1, max(0, rx // block)), min(bh - 1, max(0, ry // block))
    bx1 = min(bw, max(bx0 + 1, (rx + rw + block - 1) // block))
    by1 = min(bh, max(by0 + 1, (ry + rh + block - 1) // block))
    inside = np.zeros(shape, dtype=bool)
    inside[by0:by1, bx0:bx1] = True
    dilated = cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring = dilated & ~inside
    reference = ~dilated
    if not np.any(reference):
        reference = ~inside
    return inside, ring, reference


def _dual_working_halves(
    source: np.ndarray, working_shape: tuple[int, int], native_split: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize each sensor independently, including unequal capture widths."""
    height, width = working_shape
    native_mid = source.shape[1] // 2 if native_split is None else native_split
    working_mid = max(1, min(width - 1, round(native_mid * width / source.shape[1])))
    halves = (source[:, :native_mid], source[:, native_mid:])
    sizes = ((working_mid, height), (width - working_mid, height))
    return tuple(
        half if half.shape[:2] == (size[1], size[0]) else
        cv2.resize(half, size, interpolation=cv2.INTER_AREA)
        for half, size in zip(halves, sizes, strict=True)
    )


def _clip_half_roi(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x, y, rw, rh = rect
    x0, y0 = max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))
    x1 = min(width, max(x0 + 1, int(x + rw)))
    y1 = min(height, max(y0 + 1, int(y + rh)))
    return x0, y0, x1 - x0, y1 - y0


def _cue_map_to_doc(
    grid: np.ndarray, doc: ImageDocument, side: str | None,
    working_shape: tuple[int, int], interpolation: int, block_size: int,
) -> np.ndarray:
    # Interpolate within the selected sensor, then place it. Resizing a padded
    # full-frame heatmap would bleed nonzero values across the camera boundary.
    height, width = working_shape
    covered_h, covered_w = grid.shape[0] * block_size, grid.shape[1] * block_size
    pixels = np.zeros((height, width), dtype=grid.dtype)
    pixels[:covered_h, :covered_w] = cv2.resize(
        grid, (covered_w, covered_h), interpolation=interpolation
    )
    if side is None:
        return upscale_to_doc(pixels, doc, interpolation)
    mid = _capture_split(doc)
    assert mid is not None
    x0, x1 = (0, mid) if side == "left" else (mid, doc.width)
    placed = np.zeros((doc.height, doc.width), dtype=grid.dtype)
    placed[:, x0:x1] = cv2.resize(
        pixels, (x1 - x0, doc.height), interpolation=interpolation
    )
    return placed


def _roi_to_doc(
    rect: tuple[int, int, int, int], doc: ImageDocument,
    side: str | None, working_shape: tuple[int, int],
) -> list[int]:
    """Map a local working ROI to native pixels without crossing sensors."""
    h, w = working_shape
    split = _capture_split(doc)
    start, end = (0, doc.width)
    if side is not None and split is not None:
        start, end = (0, split) if side == "left" else (split, doc.width)
    x, y, rw, rh = rect
    x0, x1 = round(x * (end - start) / w), round((x + rw) * (end - start) / w)
    y0, y1 = round(y * doc.height / h), round((y + rh) * doc.height / h)
    return [start + x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


class DeepfakeCues:
    """Detector khám phá dấu hiệu Deepfake / AI với 6 trụ cột xử lý ảnh cổ điển."""

    name = "deepfake_cues"

    def analyze(
        self,
        doc: ImageDocument,
        cfg: ThresholdSet,
        cancel: threading.Event | None = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        lim = [
            "tín hiệu khám phá từ 6 trụ cột xử lý ảnh cổ điển — không dùng mô hình Deep Learning",
            "da mịn tự nhiên, ánh sáng yếu hoặc nén mạng xã hội mạnh có thể gây dao động điểm số",
            "signal khám phá — cần kiểm tra đối chiếu với ngữ cảnh và detector khác",
            "chỉ phân tích ROI được chọn; không xác nhận các khuôn mặt còn lại hoặc ảnh AI toàn cảnh",
        ]

        thresh = _setting(cfg, "score_thresh", 0.35)
        min_face_side = _setting(cfg, "min_face_side", 64, 16, 512)
        min_reference_blocks = _setting(cfg, "min_reference_blocks", 4, 1, 4096)
        min_roi_contrast = _setting(cfg, "min_roi_contrast", 8.0, 0, 255)
        support_thresholds = {
            key: _setting(cfg, key, default) for key, default in (
                ("smooth_support", 0.45), ("region_support", 0.20),
                ("lbp_support", 0.18), ("freq_support", 0.55),
                ("seam_support", 0.65), ("hough_support", 0.70),
            )
        }
        native_split = _capture_split(doc)
        params_hash = cfg.snapshot_hash()

        work, scale = aux_working_copy(doc, cfg)
        aux_lim = aux_limitation(scale)
        if aux_lim:
            lim.append(aux_lim)

        k_eff = scaled_kernel(int(cfg.value("noise", "median_k")), scale)
        k_eff = 5 if k_eff >= 5 else 3  # cv2.medianBlur float32 chỉ nhận k=3 hoặc 5
        bs_eff = scaled_block(int(cfg.value("noise", "block")), scale)
        min_comp = max(2, int(cfg.value("noise", "min_region_blocks")) // 2)

        gray_u8 = (
            work.array
            if work.array.ndim == 2
            else cv2.cvtColor(work.array, cv2.COLOR_BGR2GRAY)
        )
        gray = gray_u8.astype(np.float32)
        h, w = gray.shape
        bh, bw = h // bs_eff, w // bs_eff
        if bh < 4 or bw < 4:
            return self._done(
                Verdict.NOT_APPLICABLE,
                None,
                t0,
                [f"ảnh quá nhỏ ({w}x{h}) để chia lưới block {bs_eff}px"] + lim,
                evidence={"cue_unavailable": True, "unavailable_reason": "image_too_small"},
                params_hash=params_hash,
            )

        if cancelled(cancel):
            return self._done(
                Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim,
                evidence={"cue_unavailable": True, "unavailable_reason": "cancelled"},
                params_hash=params_hash,
            )

        # The layout comes from acquisition metadata or an explicit caller.
        # Panoramas and ultrawide webcams must never be split automatically.
        dual_camera_mode = native_split is not None
        dual_sigma_left: float | None = None
        dual_sigma_right: float | None = None
        dual_roi_side: str | None = None
        roi_offset_x = 0

        if dual_camera_mode:
            mid_x = max(1, min(w - 1, round(native_split * w / doc.width)))
            if min(mid_x, w - mid_x) // bs_eff < 4:
                return self._done(
                    Verdict.NOT_APPLICABLE, None, t0,
                    ["một vùng camera quá nhỏ để đo texture/phổ"] + lim,
                    evidence={"cue_unavailable": True, "unavailable_reason": "image_too_small",
                              "dual_camera_mode": True}, params_hash=params_hash,
                )
            left, right = _dual_working_halves(doc.array, (h, w), native_split)
            gray_l = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = right if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            sigma_l = _sigma_grid(gray_l.astype(np.float32), k_eff, bs_eff)
            sigma_r = _sigma_grid(gray_r.astype(np.float32), k_eff, bs_eff)
            rect_l, face_l, src_l = _detect_face_roi(
                left, gray_l, bs_eff, cfg
            )
            rect_r, face_r, src_r = _detect_face_roi(
                right, gray_r, bs_eff, cfg
            )
            rect_l = _clip_half_roi(rect_l, gray_l.shape)
            rect_r = _clip_half_roi(rect_r, gray_r.shape)
            sig_in_l, _, drop_l = _roi_sigma_stats(sigma_l, rect_l, bs_eff)
            sig_in_r, _, drop_r = _roi_sigma_stats(sigma_r, rect_r, bs_eff)
            dual_sigma_left = round(sig_in_l, 3)
            dual_sigma_right = round(sig_in_r, 3)

            def usable_face(rect, detected, candidate_gray, candidate_sigma):
                x, y, fw, fh = rect
                if not detected or min(fw, fh) < min_face_side:
                    return False
                crop = candidate_gray[y:y + fh, x:x + fw]
                contrast = float(np.percentile(crop, 95) - np.percentile(crop, 5))
                _, _, reference = _roi_masks(candidate_sigma.shape, rect, bs_eff)
                return contrast >= min_roi_contrast and np.count_nonzero(reference) >= min_reference_blocks

            usable_l = usable_face(rect_l, face_l, gray_l, sigma_l)
            usable_r = usable_face(rect_r, face_r, gray_r, sigma_r)
            if usable_l != usable_r:
                choose_right = usable_r
            elif face_l != face_r:
                choose_right = face_r
            else:
                choose_right = drop_r > drop_l + 0.05
            if choose_right:
                dual_roi_side = "right"
                rx, ry, rw, rh = rect_r
                roi_offset_x = mid_x
                gray_u8, sigma = gray_r, sigma_r
                face_detected = face_r
                roi_source = f"dual_right:{src_r}"
            else:
                dual_roi_side = "left"
                rx, ry, rw, rh = rect_l
                gray_u8, sigma = gray_l, sigma_l
                face_detected = face_l
                roi_source = f"dual_left:{src_l}"
            # Every remaining filter, reference statistic and Otsu estimate is
            # local to this half. Its block grid starts at its own x=0.
            gray = gray_u8.astype(np.float32)
            h, w = gray.shape
            bh, bw = sigma.shape
            lim.append(
                f"chế độ Dual-Camera Cross-Sensor: σ_L={sig_in_l:.2f} vs σ_R={sig_in_r:.2f} → chọn nửa {dual_roi_side}"
            )
        else:
            sigma = _sigma_grid(gray, k_eff, bs_eff)
            (rx, ry, rw, rh), face_detected, roi_source = _detect_face_roi(
                work.array, gray_u8, bs_eff, cfg
            )
            rx, ry, rw, rh = _clip_half_roi((rx, ry, rw, rh), gray_u8.shape)

        roi_rect = _roi_to_doc((rx, ry, rw, rh), doc, dual_roi_side, (h, w))
        base_evidence = {
            "roi_source": roi_source,
            "dual_camera_mode": dual_camera_mode,
            "dual_sigma_left": dual_sigma_left,
            "dual_sigma_right": dual_sigma_right,
            "dual_roi_side": dual_roi_side,
            "capture_layout": doc.capture_layout,
            "capture_split_x": native_split,
            "face_detected": bool(face_detected),
            "roi_rect": roi_rect,
            "roi_rect_working": [rx + roi_offset_x, ry, rw, rh],
            "roi_coordinate_space": "original_pixels",
            "block": bs_eff,
            "scale": round(scale, 4),
            "decision_threshold": thresh,
        }
        unavailable_reason = ""

        if not face_detected:
            unavailable_reason = "no_face"
            lim.insert(
                0,
                "Chưa định vị được vùng mặt trên khung này. "
                "Không cấp điểm Deepfake; các số đo vùng trung tâm chỉ dùng minh họa xử lý ảnh.",
            )
        elif min(rw, rh) < min_face_side:
            unavailable_reason = "roi_too_small"
            lim.insert(0, f"vùng mặt {rw}x{rh}px nhỏ hơn mức tối thiểu {min_face_side:g}px")
        elif "skin_haar_dip" in roi_source:
            lim.append("ROI từ màu da và cấu trúc sáng tối là ứng viên mặt; có thể nhầm vật thể cùng màu")

        inside_roi, seam_ring, outer_bg = _roi_masks((bh, bw), (rx, ry, rw, rh), bs_eff)
        if np.count_nonzero(outer_bg) < min_reference_blocks:
            lim.insert(
                0,
                "không đủ nền cùng cảm biến ngoài ROI để đối chiếu "
                "— không dùng chính ROI hoặc camera còn lại làm nền"
            )
            return self._done(
                Verdict.INCONCLUSIVE,
                None,
                t0,
                lim,
                evidence={
                    **base_evidence,
                    "cue_unavailable": True,
                    "unavailable_reason": "insufficient_background",
                    "reference_blocks": int(np.count_nonzero(outer_bg)),
                },
                params_hash=params_hash,
            )

        med_out = float(np.median(sigma[outer_bg]))
        med_in = float(np.median(sigma[inside_roi]))
        mad_out = float(np.median(np.abs(sigma[outer_bg] - med_out))) + 1e-6

        if med_out > 1.0:
            smooth_cue = float(np.clip((med_out - med_in) / (med_out + 1e-6), 0.0, 1.0))
            low_roi = (inside_roi & (sigma < 0.65 * med_out)).astype(np.uint8)
        else:
            smooth_cue = 0.0
            low_roi = np.zeros((bh, bw), dtype=np.uint8)

        if np.any(seam_ring):
            seam_std = float(np.std(sigma[seam_ring]))
            seam_jump = abs(float(np.median(sigma[seam_ring])) - med_in)
            seam_raw = (0.65 * seam_std + 0.35 * seam_jump) / (
                2.5 * 1.4826 * mad_out + med_out * 0.25 + 1e-6
            )
            seam_cue = float(np.clip(seam_raw, 0.0, 1.0))
        else:
            seam_cue = 0.0

        if cancelled(cancel):
            return self._done(
                Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim,
                evidence={**base_evidence, "cue_unavailable": True, "unavailable_reason": "cancelled"},
                params_hash=params_hash,
            )

        # Trụ cột 3: Vi kết cấu LBP 8 lân cận & khoảng cách χ² ROI vs nền (C6).
        lbp_cue, lbp_chi2, flat_in, flat_out, flat_blocks = _lbp_texture_cues(
            gray_u8, inside_roi, outer_bg, bs_eff
        )

        # Trụ cột 4: Phổ tần số 2D FFT (đỉnh checkerboard & suy hao cao tần hướng tâm).
        roi_f = gray[ry : ry + rh, rx : rx + rw]
        roi_u8 = gray_u8[ry : ry + rh, rx : rx + rw]
        contrast = float(np.percentile(roi_u8, 95) - np.percentile(roi_u8, 5))
        if not unavailable_reason and contrast < min_roi_contrast:
            unavailable_reason = "low_contrast"
            lim.insert(0, "vùng mặt thiếu tương phản; không đủ dữ liệu để cấp điểm Deepfake")
        freq_cue, peak_ratio, spectral_rolloff, hf_conc8 = _fft_highfreq_cue(roi_f)

        # Trụ cột 5 & 6: Biên ghép hình học (Canny + Boundary + Hough) & đối xứng HOG/Hu.
        seam_hough_cue, seam_hough_lines = _seam_canny_hough_cue(roi_u8)
        sym_cue, hog_asym = _facial_hog_symmetry_cue(roi_u8)

        # Kết hợp Otsu trên bản đồ suy giảm nhiễu + thành phần liên thông 8-lân cận.
        sigma_drop = (
            np.clip((med_out - sigma) / (med_out + 1e-6), 0.0, 1.0)
            if med_out > 1.0
            else np.zeros((bh, bw), dtype=np.float32)
        )
        otsu_t = int(threshold.otsu_threshold(np.clip(sigma_drop * 255.0, 0, 255).astype(np.uint8)))
        if med_out > 1.0 and otsu_t > 0:
            otsu_roi = (inside_roi & ((sigma_drop * 255.0) >= max(float(otsu_t), 85.0))).astype(np.uint8)
            low_roi = np.maximum(low_roi, otsu_roi)

        labels_cc, n_cc = morph_algos.connected_components((low_roi * 255).astype(np.uint8), connectivity=8)
        areas = np.bincount(labels_cc.ravel(), minlength=n_cc + 1)
        keep = np.zeros_like(low_roi)
        kept_area = 0
        for comp_id in range(1, n_cc + 1):
            area = int(areas[comp_id])
            if area >= min_comp:
                keep[labels_cc == comp_id] = 1
                kept_area += area

        roi_flag_frac = kept_area / max(int(inside_roi.sum()), 1)
        raw_score = float(
            np.clip(
                0.34 * smooth_cue
                + 0.18 * min(1.0, roi_flag_frac * 2.5)
                + 0.16 * lbp_cue
                + 0.12 * seam_cue
                + 0.12 * freq_cue
                + 0.05 * seam_hough_cue
                + 0.03 * sym_cue,
                0.0,
                1.0,
            )
        )

        # Smoothing, LBP flattening and low-noise area describe the same family
        # of effects. Counting all three as separate votes overstates support.
        # Require at least two cue families; these are still uncalibrated and
        # correlated observations, not independent evidence of a fake face.
        texture_support = (
            smooth_cue >= support_thresholds["smooth_support"]
            and roi_flag_frac >= support_thresholds["region_support"]
        ) or lbp_cue >= support_thresholds["lbp_support"]
        spectral_support = freq_cue >= support_thresholds["freq_support"]
        boundary_support = (
            seam_cue >= support_thresholds["seam_support"]
            or seam_hough_cue >= support_thresholds["hough_support"]
        )
        support = {
            "texture": bool(texture_support), "frequency": bool(spectral_support),
            "boundary": bool(boundary_support),
        }
        corroborated = sum(support.values()) >= 2
        score = None if unavailable_reason else round(raw_score, 4)

        # Heatmap tổng hợp đa lớp: suy giảm nhiễu σ + mật độ phẳng LBP trong ROI.
        roi_weight = 0.30 + 0.70 * inside_roi.astype(np.float32)
        lbp_excess = np.clip((flat_blocks - flat_out) / max(1.0 - flat_out, 0.25), 0.0, 1.0)
        heat_blocks = (0.72 * sigma_drop + 0.28 * lbp_excess) * roi_weight
        heat_img = _cue_map_to_doc(
            np.clip(heat_blocks, 0.0, 1.0), doc, dual_roi_side, (h, w), cv2.INTER_LINEAR, bs_eff
        ) if not unavailable_reason else None

        has_flag = bool(not unavailable_reason and corroborated and raw_score >= thresh)
        if has_flag and kept_area == 0:
            # Nếu cờ kích hoạt từ phổ FFT/biên seam mà mask nhiễu block trống, tô vùng ROI.
            keep[inside_roi] = 1
        mask = (
            _cue_map_to_doc(
                (keep * 255).astype(np.uint8), doc, dual_roi_side, (h, w), cv2.INTER_NEAREST, bs_eff
            )
            if has_flag
            else None
        )
        verdict = Verdict.INCONCLUSIVE if has_flag or unavailable_reason else Verdict.NO_INDICATION
        if has_flag:
            lim.insert(
                0 if face_detected else 1,
                f"phát hiện bất thường mịn/LBP/phổ vùng ROI (score={score:.2f}, "
                f"σ_in={med_in:.2f} vs σ_out={med_out:.2f}, lbp_χ²={lbp_chi2:.3f}) — cần đối chiếu thêm"
            )

        return self._done(
            verdict,
            score,
            t0,
            lim,
            mask=mask,
            heatmap=heat_img,
            evidence={
                **base_evidence,
                "corroborated": corroborated,
                "cue_flag": has_flag,
                "cue_unavailable": bool(unavailable_reason),
                "unavailable_reason": unavailable_reason or None,
                "support_families": [name for name, active in support.items() if active],
                "reference_blocks": int(np.count_nonzero(outer_bg)),
                "roi_contrast": round(contrast, 3),
                "sigma_in_median": round(med_in, 3),
                "sigma_out_median": round(med_out, 3),
                "smooth_cue": round(smooth_cue, 4),
                "seam_cue": round(seam_cue, 4),
                "lbp_cue": round(lbp_cue, 4),
                "lbp_chi2": round(lbp_chi2, 4),
                "lbp_flat_in": round(flat_in, 4),
                "lbp_flat_out": round(flat_out, 4),
                "freq_cue": round(freq_cue, 4),
                "fft_hf_peak_ratio": round(peak_ratio, 3),
                "spectral_rolloff": round(spectral_rolloff, 3),
                "fft_hf_conc8": round(hf_conc8, 5),
                "seam_hough_cue": round(seam_hough_cue, 4),
                "seam_hough_lines": seam_hough_lines,
                "symmetry_cue": round(sym_cue, 4),
                "hog_asym": round(hog_asym, 4),
                "kept_blocks": kept_area,
                "roi_flag_fraction": round(roi_flag_frac, 4),
            },
            params_hash=params_hash,
        )

    def _done(
        self,
        verdict: Verdict,
        score: float | None,
        t0: float,
        limitations: list[str],
        evidence: dict | None = None,
        mask: np.ndarray | None = None,
        heatmap: np.ndarray | None = None,
        params_hash: str = "",
    ) -> AlgorithmResult:
        _cascade, haar_status = _load_haar_cascade()
        limitations = list(limitations)
        if haar_status != "available":
            limitations.insert(
                min(1, len(limitations)),
                "OpenCV không cung cấp Haar Cascade dùng được; chỉ còn định vị màu da/cấu trúc "
                "thận trọng, có thể bỏ sót vùng mặt. Cài môi trường OpenCV 4 theo pyproject.toml.",
            )
        return AlgorithmResult(
            method=MethodName.DEEPFAKE_CUES,
            verdict=verdict,
            score=score,
            score_definition=SCORE_DEF,
            mask=mask,
            heatmap=heatmap,
            evidence={
                "variant": "dip_6pillar_deepfake_cues_v3",
                "haar_cascade_status": haar_status,
                "face_detector_backend": (
                    "haar_with_skin_fallback" if haar_status == "available" else "skin_fallback_only"
                ),
                "calibration_status": "uncalibrated",
                "score_is_probability": False,
                "cue_flag": False,
                "cue_unavailable": False,
                "unavailable_reason": None,
                **(evidence or {}),
            },
            limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
