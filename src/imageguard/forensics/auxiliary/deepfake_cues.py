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

Hỗ trợ Dual-Camera Cross-Sensor khi nhận ảnh ghép từ 2 camera (W ≥ 2.1·H).

Contract bắt buộc: thuộc ``AUX_METHODS`` — chỉ trả ``INCONCLUSIVE``,
``NO_INDICATION`` hoặc ``NOT_APPLICABLE``; KHÔNG BAO GIỜ trả ``SUSPICIOUS_INDICATION``.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
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
    "LBP χ², phổ 2D FFT, Canny-Hough biên ghép & đối xứng HOG/Hu) — tín hiệu khám phá Deepfake/AI"
)

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
    """Kiểm chứng ứng viên khuôn mặt bằng Haar two_v (dải mắt/má) & đối xứng trái-phải (C6)."""
    roi = gray_u8[y : y + rh, x : x + rw]
    if roi.shape[0] < 24 or roi.shape[1] < 24:
        return False

    # Chuẩn hóa kích thước trung gian nhỏ để đo đối xứng cấu trúc ổn định và cực nhanh.
    norm = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    integ = cv2.integral(norm)

    def _box_mean(x0: int, y0: int, x1: int, y1: int) -> float:
        area = float(max(1, (x1 - x0) * (y1 - y0)))
        s = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        return float(s / area)

    # Dải mắt (18%..44% chiều cao, lõi ngang) vs dải má/mũi dưới (48%..74% chiều cao).
    eye_mean = _box_mean(8, 9, 40, 21)
    cheek_mean = _box_mean(8, 23, 40, 36)
    haar_two_v = abs(cheek_mean - eye_mean)
    eye_std = float(np.std(norm[9:22, 8:40]))

    # Đối xứng gương trái-phải sau khi trừ trung bình sáng để chịu ánh sáng lệch nhẹ.
    left = norm[:, :24]
    right = norm[:, 24:][:, ::-1]
    left_c = left - float(np.mean(left))
    right_c = right - float(np.mean(right))
    l1_sym = 1.0 - float(np.mean(np.abs(left_c - right_c))) / 128.0

    # Khuôn mặt thật hoặc mô phỏng có chi tiết vùng mắt + tương phản Haar + đối xứng trái-phải.
    return (haar_two_v >= 2.5 or eye_std >= 4.0) and l1_sym >= 0.55


def _detect_face_roi(
    work_arr: np.ndarray,
    gray_u8: np.ndarray,
    block_size: int,
) -> tuple[tuple[int, int, int, int], bool, str]:
    """Định vị ROI khuôn mặt 3 tầng: Haar Cascade -> YCrCb/HSV Skin + SAT Haar -> Central."""
    h, w = gray_u8.shape
    min_dim = max(32, block_size * 2)

    # Tầng 1: OpenCV Haar Cascade (nếu bản build hỗ trợ và tìm được file XML).
    haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if haar_dir and cascade_cls is not None:
        cascade = cascade_cls(f"{haar_dir}haarcascade_frontalface_default.xml")
        if not cascade.empty():
            faces = cascade.detectMultiScale(
                gray_u8,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(min_dim, min_dim),
            )
            if len(faces) > 0:
                x, y, rw, rh = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
                return (int(x), int(y), int(rw), int(rh)), True, "haar_cascade"

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
        best_area = 0
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
            if _verify_face_candidate_sat(gray_u8, x, y, rw, rh) and area > best_area:
                best_area = area
                best_rect = (x, y, rw, rh)

        if best_rect is not None:
            return best_rect, True, "skin_haar_dip"

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


def _fft_highfreq_cue(roi_gray: np.ndarray) -> tuple[float, float, float]:
    """Đo đỉnh tuần hoàn (magnitude p99.8/median) & suy hao phổ 2D FFT (C3)."""
    rh, rw = roi_gray.shape
    if rh < 16 or rw < 16:
        return 0.0, 1.0, 1.0
    centered = roi_gray.astype(np.float32) - float(roi_gray.mean())
    win = np.outer(np.hanning(rh), np.hanning(rw)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(centered * win)))
    spec = np.log1p(mag)

    yy, xx = np.ogrid[:rh, :rw]
    cy, cx = (rh - 1) / 2.0, (rw - 1) / 2.0
    norm_r = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    hf_mask = (norm_r >= 0.45) & (norm_r <= 0.95)
    mf_mask = (norm_r >= 0.15) & (norm_r < 0.45)
    if not np.any(hf_mask):
        return 0.0, 1.0, 1.0

    # Đỉnh tuần hoàn checkerboard đo trên biên độ tuyến tính (tránh bị log1p nén dải đỉnh).
    hf_mag = mag[hf_mask]
    hf_med = float(np.median(hf_mag)) + 1e-6
    hf_p998 = float(np.percentile(hf_mag, 99.8))
    peak_ratio = hf_p998 / hf_med

    hf_mean = float(np.mean(spec[hf_mask]))
    mf_mean = float(np.mean(spec[mf_mask])) if np.any(mf_mask) else hf_mean
    spectral_rolloff = hf_mean / (mf_mean + 1e-6)

    peak_score = float(np.clip((peak_ratio - 3.3) / 2.2, 0.0, 1.0))
    rolloff_score = float(np.clip((0.58 - spectral_rolloff) / 0.25, 0.0, 1.0))
    freq_cue = float(np.clip(max(peak_score, 0.65 * peak_score + 0.35 * rolloff_score), 0.0, 1.0))
    return freq_cue, peak_ratio, spectral_rolloff


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


def _dual_working_halves(
    source: np.ndarray, working_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Resize each sensor independently, including odd native/working widths."""
    height, width = working_shape
    native_mid, working_mid = source.shape[1] // 2, width // 2
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
    working_shape: tuple[int, int], interpolation: int,
) -> np.ndarray:
    if side is None:
        return upscale_to_doc(grid, doc, interpolation)
    # Interpolate within the selected sensor, then place it. Resizing a padded
    # full-frame heatmap would bleed nonzero values across the camera boundary.
    height, width = working_shape
    pixels = cv2.resize(grid, (width, height), interpolation=interpolation)
    mid = doc.width // 2
    x0, x1 = (0, mid) if side == "left" else (mid, doc.width)
    placed = np.zeros((doc.height, doc.width), dtype=grid.dtype)
    placed[:, x0:x1] = cv2.resize(
        pixels, (x1 - x0, doc.height), interpolation=interpolation
    )
    return placed


class DeepfakeCues:
    """Detector khám phá dấu hiệu Deepfake / AI với 4 trụ cột Xử lý ảnh cổ điển."""

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
        ]

        work, scale = aux_working_copy(doc, cfg)
        aux_lim = aux_limitation(scale)
        if aux_lim:
            lim.append(aux_lim)

        k_eff = scaled_kernel(int(cfg.value("noise", "median_k")), scale)
        k_eff = 5 if k_eff >= 5 else 3  # cv2.medianBlur float32 chỉ nhận k=3 hoặc 5
        bs_eff = scaled_block(int(cfg.value("noise", "block")), scale)
        min_comp = max(2, int(cfg.value("noise", "min_region_blocks")) // 2)
        df_overrides = cfg.raw.get("deepfake", {}) if isinstance(cfg.raw, dict) else {}
        thresh = float(df_overrides.get("score_thresh", 0.35))

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
            )

        if cancelled(cancel):
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        # Trụ cột 1: Định vị ROI khuôn mặt — hỗ trợ Dual-Camera Cross-Sensor nếu khung ghép đôi (W >= 2.1*H).
        dual_camera_mode = w >= int(2.1 * h) and bw >= 8
        dual_sigma_left: float | None = None
        dual_sigma_right: float | None = None
        dual_roi_side: str | None = None
        roi_offset_x = 0

        if dual_camera_mode:
            mid_x = w // 2
            left, right = _dual_working_halves(doc.array, (h, w))
            gray_l = left if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_r = right if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            sigma_l = _sigma_grid(gray_l.astype(np.float32), k_eff, bs_eff)
            sigma_r = _sigma_grid(gray_r.astype(np.float32), k_eff, bs_eff)
            rect_l, face_l, src_l = _detect_face_roi(
                left, gray_l, bs_eff
            )
            rect_r, face_r, src_r = _detect_face_roi(
                right, gray_r, bs_eff
            )
            rect_l = _clip_half_roi(rect_l, gray_l.shape)
            rect_r = _clip_half_roi(rect_r, gray_r.shape)
            sig_in_l, _, drop_l = _roi_sigma_stats(sigma_l, rect_l, bs_eff)
            sig_in_r, _, drop_r = _roi_sigma_stats(sigma_r, rect_r, bs_eff)
            dual_sigma_left = round(sig_in_l, 3)
            dual_sigma_right = round(sig_in_r, 3)

            choose_right = (drop_r > drop_l + 0.05) or (
                abs(drop_r - drop_l) <= 0.05 and face_r and not face_l
            )
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
                work.array, gray_u8, bs_eff
            )

        if not face_detected:
            lim.append(
                "không phát hiện khuôn mặt frontal — đang kiểm tra vùng trung tâm với ngưỡng đồng thuận đa tín hiệu"
            )

        bx0 = min(bw - 1, max(0, rx // bs_eff))
        by0 = min(bh - 1, max(0, ry // bs_eff))
        bx1 = min(bw, max(bx0 + 1, (rx + rw + bs_eff - 1) // bs_eff))
        by1 = min(bh, max(by0 + 1, (ry + rh + bs_eff - 1) // bs_eff))
        inside_roi = np.zeros((bh, bw), dtype=bool)
        inside_roi[by0:by1, bx0:bx1] = True

        dilated_roi = cv2.dilate(inside_roi.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        seam_ring = dilated_roi & (~inside_roi)
        outer_bg = ~dilated_roi
        if not np.any(outer_bg):
            outer_bg = ~inside_roi
        if dual_camera_mode and not np.any(outer_bg):
            lim.append(
                "không đủ nền cùng cảm biến trong nửa camera đã chọn "
                "(ROI chiếm gần toàn bộ nửa) — không mượn nền camera còn lại"
            )
            return self._done(
                Verdict.INCONCLUSIVE,
                None,
                t0,
                lim,
                evidence={
                    "roi_source": roi_source,
                    "dual_camera_mode": dual_camera_mode,
                    "dual_sigma_left": dual_sigma_left,
                    "dual_sigma_right": dual_sigma_right,
                    "dual_roi_side": dual_roi_side,
                    "face_detected": face_detected,
                    "roi_rect": [rx + roi_offset_x, ry, rw, rh],
                    "block": bs_eff,
                },
            )
        if not np.any(outer_bg):
            outer_bg = np.ones((bh, bw), dtype=bool)

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
            return self._done(Verdict.INCONCLUSIVE, None, t0, ["cancelled"] + lim)

        # Trụ cột 3: Vi kết cấu LBP 8 lân cận & khoảng cách χ² ROI vs nền (C6).
        lbp_cue, lbp_chi2, flat_in, flat_out, flat_blocks = _lbp_texture_cues(
            gray_u8, inside_roi, outer_bg, bs_eff
        )

        # Trụ cột 4: Phổ tần số 2D FFT (đỉnh checkerboard & suy hao cao tần hướng tâm).
        roi_f = gray[ry : ry + rh, rx : rx + rw]
        roi_u8 = gray_u8[ry : ry + rh, rx : rx + rw]
        freq_cue, peak_ratio, spectral_rolloff = _fft_highfreq_cue(roi_f)

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

        # Chống báo giả cho cảnh nền tự nhiên không có mặt (ví dụ bàn/tường nhẵn ở camera phụ):
        # nếu chỉ là fallback trung tâm (không thấy mặt), đòi hỏi đồng thuận nhiều tín hiệu.
        corroborated = True
        if not face_detected:
            corroborated = (
                (smooth_cue >= 0.45 and roi_flag_frac >= 0.20 and (lbp_cue >= 0.18 or seam_cue >= 0.30))
                or freq_cue >= 0.55
                or seam_cue >= 0.65
                or seam_hough_cue >= 0.70
            )
            if not corroborated:
                raw_score *= 0.55

        score = round(raw_score, 4)

        # Heatmap tổng hợp đa lớp: suy giảm nhiễu σ + mật độ phẳng LBP trong ROI.
        roi_weight = 0.30 + 0.70 * inside_roi.astype(np.float32)
        lbp_excess = np.clip((flat_blocks - flat_out) / max(1.0 - flat_out, 0.25), 0.0, 1.0)
        heat_blocks = (0.72 * sigma_drop + 0.28 * lbp_excess) * roi_weight
        heat_img = _cue_map_to_doc(
            np.clip(heat_blocks, 0.0, 1.0), doc, dual_roi_side, (h, w), cv2.INTER_LINEAR
        )

        has_flag = score >= thresh and (
            kept_area > 0 or freq_cue >= 0.55 or seam_cue >= 0.65 or seam_hough_cue >= 0.70
        )
        if has_flag and kept_area == 0:
            # Nếu cờ kích hoạt từ phổ FFT/biên seam mà mask nhiễu block trống, tô vùng ROI.
            keep[inside_roi] = 1
        mask = (
            _cue_map_to_doc((keep * 255).astype(np.uint8), doc, dual_roi_side, (h, w), cv2.INTER_NEAREST)
            if has_flag
            else None
        )
        verdict = Verdict.INCONCLUSIVE if has_flag else Verdict.NO_INDICATION
        if verdict == Verdict.INCONCLUSIVE:
            lim.append(
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
                "roi_source": roi_source,
                "dual_camera_mode": dual_camera_mode,
                "dual_sigma_left": dual_sigma_left,
                "dual_sigma_right": dual_sigma_right,
                "dual_roi_side": dual_roi_side,
                "face_detected": face_detected,
                "corroborated": corroborated,
                "roi_rect": [rx + roi_offset_x, ry, rw, rh],
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
                "seam_hough_cue": round(seam_hough_cue, 4),
                "seam_hough_lines": seam_hough_lines,
                "hog_asym": round(hog_asym, 4),
                "kept_blocks": kept_area,
                "block": bs_eff,
                "scale": round(scale, 4),
            },
            params_hash=cfg.snapshot_hash(),
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
        return AlgorithmResult(
            method=MethodName.DEEPFAKE_CUES,
            verdict=verdict,
            score=score,
            score_definition=SCORE_DEF,
            mask=mask,
            heatmap=heatmap,
            evidence={"variant": "dip_4pillar_deepfake_cues", **(evidence or {})},
            limitations=limitations,
            params_hash=params_hash,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
