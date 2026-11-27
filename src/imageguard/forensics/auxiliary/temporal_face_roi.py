"""Cue thời gian: vùng mặt trong burst khung có đứng yên bất thường không.

Operator chụp burst 4–6 khung từ một camera. ROI mặt gần như không đổi có
thể do người đang đứng yên, camera ít nhiễu, ảnh tĩnh hoặc khung đóng băng.
Đây là số đo chất lượng thu nhận, không phải bằng chứng Deepfake hay kiểm
tra liveness. Vì vậy cue thời gian không thay đổi score/verdict của ảnh.

Cách đo (``temporal_static_cue``):

1. ``_detect_face_roi`` của ``deepfake_cues`` chạy trên KHUNG GIỮA burst để
   chọn ROI — cùng detector 3 tầng với pipeline deepfake nên vùng đo nhất quán.
2. Trong ROI đó, với mỗi khung lấy ảnh xám (luma) và đáp ứng Laplacian
   (``cv2.Laplacian``, ksize=1 mặc định) rồi tính phương sai thời gian theo từng pixel:
   ``mean_pix Var_t(luma)`` và ``mean_pix Var_t(laplacian)``.
3. Hai đại lượng chuẩn hoá về [0,1] bằng span cố định (hiệu chuẩn trên cohort
   synthetic — xem ``tests/test_temporal_face_roi.py``) rồi trộn đều:
   ``temporal_static_score = 1 − change`` ∈ [0,1]. Khung giống hệt nhau → 0
   biến thiên → score 1.0; nhiễu độc lập từng khung hoặc chuyển động → score
   thấp.

``apply_temporal_cue`` bổ sung số đo, trạng thái và lý do vào evidence;
score/verdict/score_definition gốc được giữ nguyên. Evidence không chứa
đường dẫn hay dữ liệu định danh.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from imageguard.core.models import AlgorithmResult
from imageguard.forensics.auxiliary.deepfake_cues import _detect_face_roi

# Ngưỡng mô tả độ tĩnh, không phải ngưỡng phân loại Deepfake. Giá trị chọn
# trên cohort synthetic của tests/test_temporal_face_roi.py (khung giống
# hệt → 1.0; nhiễu iid σ≈12 → ~0.66; nhiễu cảm biến nhẹ σ≈3 → ~0.98; dịch
# chuyển 1px → ~0.1) — KHÔNG phải tỉ lệ chính xác trên dữ liệu thật.
TEMPORAL_STATIC_MIN = 0.85

# Giữ tên public để tương thích caller cũ; cue thời gian không cộng điểm.
TEMPORAL_SCORE_MAX_DELTA = 0.0

_BLOCK = 8  # block_size cho _detect_face_roi (min_dim = 2×8 = 32px) — như face_roi_cross_compare
_MIN_ROI_SIDE = 8  # ROI nhỏ hơn → không đủ pixel đo phương sai thời gian

# Chuẩn hoá phương sai thời gian trung bình-theo-pixel về [0,1]. Span chọn theo
# cohort synthetic 320×320 (ROI ~113×145px): nhiễu iid σ≈12 trong kênh BGR cho
# luma_tv≈51 / lap_tv≈1023; σ≈3 → 3.3 / 66.6; trượt 1px → 186 / 2307.
_LUMA_VAR_SPAN = 150.0
_LAP_VAR_SPAN = 3000.0
_W_LUMA = 0.5
_W_LAP = 0.5


def _gray_u8_f32(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Trả (gray float32, bản BGR để detect). Hỗ trợ HxWx3 BGR và HxW gray."""
    if arr.ndim == 2:
        return arr.astype(np.float32), None
    if arr.ndim == 3 and arr.shape[2] >= 3:
        bgr = arr[:, :, :3]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), bgr
    raise ValueError("frames_bgr phần tử phải là ndarray HxWx3 hoặc HxW")


def _resize_to_ref(gray: np.ndarray, ref_hw: tuple[int, int]) -> np.ndarray:
    """Đồng bộ kích thước gray về (h, w) của khung giữa — phòng burst lẫn size."""
    rh, rw = ref_hw
    if gray.shape == (rh, rw):
        return gray
    interp = cv2.INTER_AREA if gray.shape[0] > rh or gray.shape[1] > rw else cv2.INTER_LINEAR
    return cv2.resize(gray, (rw, rh), interpolation=interp).astype(np.float32)


def _clip_rect(
    rect: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    x, y, rw, rh = rect
    x0, y0 = max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1))
    x1 = min(w, max(x0 + 1, int(x + rw)))
    y1 = min(h, max(y0 + 1, int(y + rh)))
    return x0, y0, x1 - x0, y1 - y0


def _neutral_dict(frame_count: int) -> dict:
    return {
        "temporal_static_score": 0.0,
        "face_detected": False,
        "roi_source": "insufficient_frames",
        "frame_count": int(frame_count),
        "roi_rect": None,
        "luma_temporal_var": 0.0,
        "laplacian_temporal_var": 0.0,
    }


def temporal_static_cue(frames_bgr: list[np.ndarray]) -> dict:
    """Đo mức 'đứng yên' của ROI mặt qua chuỗi khung burst → dict số đo.

    ``temporal_static_score`` ∈ [0,1], càng cao mặt càng tĩnh. Cần ≥2 khung;
    ít hơn → score 0.0 (không kết luận được tĩnh). ROI chọn trên khung giữa;
    ``face_detected`` báo ROI đó có phải mặt thật hay fallback trung tâm.
    """
    frames = list(frames_bgr) if frames_bgr is not None else []
    if len(frames) < 2:
        return _neutral_dict(len(frames))

    grays: list[np.ndarray] = []
    mid_work: np.ndarray | None = None
    mid_idx = len(frames) // 2
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        gray, bgr = _gray_u8_f32(arr)
        grays.append(gray)
        if i == mid_idx:
            # work_arr cho detector: BGR khi có, ngược lại gray 2D — detector
            # tự rơi về fallback trung tâm (face_detected=False) khi thiếu màu.
            mid_work = bgr if bgr is not None else grays[mid_idx]
    if mid_work is None:  # pragma: no cover - mid_idx luôn hợp lệ khi len>=2
        return _neutral_dict(len(frames))

    ref_hw = grays[mid_idx].shape
    for i, gray in enumerate(grays):
        grays[i] = _resize_to_ref(gray, ref_hw)

    mid_gray_u8 = grays[mid_idx].clip(0, 255).astype(np.uint8)
    (rx, ry, rw, rh), face_detected, roi_source = _detect_face_roi(
        mid_work, mid_gray_u8, _BLOCK
    )
    rx, ry, rw, rh = _clip_rect((rx, ry, rw, rh), ref_hw)
    if rw < _MIN_ROI_SIDE or rh < _MIN_ROI_SIDE:
        return _neutral_dict(len(frames)) | {
            "face_detected": bool(face_detected),
            "roi_source": roi_source,
            "roi_rect": [rx, ry, rw, rh],
        }

    luma_rois: list[np.ndarray] = []
    lap_rois: list[np.ndarray] = []
    for gray in grays:
        roi = gray[ry : ry + rh, rx : rx + rw]
        luma_rois.append(roi.copy())
        # Laplacian tính trên toàn khung xám rồi mới crop — tránh biên giả ở
        # mép ROI (cv2.Laplacian dùng cửa sổ 3×3).
        lap_full = cv2.Laplacian(gray, cv2.CV_32F)
        lap_rois.append(lap_full[ry : ry + rh, rx : rx + rw].copy())

    luma_stack = np.stack(luma_rois, axis=0)
    lap_stack = np.stack(lap_rois, axis=0)
    # Phương sai thời gian theo từng pixel rồi trung bình trên ROI — đại lượng
    # này bắt được cả nhiễu iid từng khung lẫn chuyển động, khác với phương sai
    # của chuỗi đặc trưng tóm tắt (vốn bị nhiễu trung bình hoá triệt tiêu).
    luma_tv = float(np.mean(np.var(luma_stack, axis=0, dtype=np.float64)))
    lap_tv = float(np.mean(np.var(lap_stack, axis=0, dtype=np.float64)))

    luma_change = float(np.clip(luma_tv / _LUMA_VAR_SPAN, 0.0, 1.0))
    lap_change = float(np.clip(lap_tv / _LAP_VAR_SPAN, 0.0, 1.0))
    change = _W_LUMA * luma_change + _W_LAP * lap_change
    static = float(np.clip(1.0 - change, 0.0, 1.0))

    return {
        "temporal_static_score": round(static, 4),
        "face_detected": bool(face_detected),
        "roi_source": roi_source,
        "frame_count": len(frames),
        "roi_rect": [rx, ry, rw, rh],
        "luma_temporal_var": round(luma_tv, 4),
        "laplacian_temporal_var": round(lap_tv, 4),
    }


def apply_temporal_cue(
    result: AlgorithmResult, frames_bgr: list[np.ndarray]
) -> AlgorithmResult:
    """Ghi độ tĩnh của burst, giữ nguyên kết quả phân tích không gian.

    Người thật có thể đứng yên và video giả có thể chuyển động; phép đo này
    không phân biệt hai trường hợp đó. Chỉ bổ sung trạng thái thu nhận và
    số đo vào evidence/limitations, không cộng vào score hoặc đổi verdict.
    """
    t0 = time.perf_counter()
    try:
        cue = temporal_static_cue(frames_bgr)
    except Exception as exc:  # noqa: BLE001 - cue phụ không được làm hỏng kết quả gốc
        cue = _neutral_dict(0)
        cue["error"] = type(exc).__name__

    limitations = list(result.limitations)
    static = float(cue["temporal_static_score"])
    evidence = dict(result.evidence)
    if cue.get("error"):
        status = "error"
        reason = (
            "bỏ qua cue thời gian burst — không đo được "
            f"({cue['error']})"
        )
    elif cue["frame_count"] < 2:
        status = "insufficient_frames"
        reason = "không đủ khung burst để đo cue thời gian"
    elif not cue["face_detected"]:
        status = "face_unavailable"
        reason = (
            f"burst {cue['frame_count']} khung: không phát hiện mặt — "
            "cue thời gian không áp dụng"
        )
    elif static >= TEMPORAL_STATIC_MIN:
        status = "static"
        reason = (
            f"vùng mặt gần như đứng yên qua {cue['frame_count']} khung "
            f"(temporal_static={static:.2f}) — người thật cũng có thể đứng yên; "
            "độ tĩnh không phải bằng chứng Deepfake và không cộng vào điểm"
        )
    else:
        status = "variable"
        reason = (
            f"vùng mặt biến thiên theo thời gian qua {cue['frame_count']} "
            f"khung (temporal_static={static:.2f}) — chuyển động không chứng minh ảnh thật"
        )
    limitations.append(reason)

    evidence.update(
        {
            "temporal_status": status,
            "temporal_reason": reason,
            "temporal_frame_count": int(cue["frame_count"]),
            "temporal_static_score": cue["temporal_static_score"],
            "temporal_luma_var": cue["luma_temporal_var"],
            "temporal_laplacian_var": cue["laplacian_temporal_var"],
            "temporal_face_detected": int(bool(cue["face_detected"])),
        }
    )

    return AlgorithmResult(
        method=result.method,
        verdict=result.verdict,
        score=result.score,
        score_definition=result.score_definition,
        mask=result.mask,
        heatmap=result.heatmap,
        evidence=evidence,
        limitations=limitations,
        params_hash=result.params_hash,
        duration_ms=result.duration_ms + int((time.perf_counter() - t0) * 1000),
    )
