"""Chuỗi xem trước dấu hiệu deepfake trên khung camera — không phụ thuộc Qt."""

from __future__ import annotations

import hashlib

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import AlgorithmResult, ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.forensics.auxiliary.temporal_face_roi import TEMPORAL_STATIC_MIN
from imageguard.ui import strings

_EVIDENCE_KEYS = (
    "face_detected",
    "roi_source",
    "cue_flag",
    "calibration_status",
    "dual_camera_mode",
    "smooth_cue",
    "sigma_in_median",
    "sigma_out_median",
    "lbp_cue",
    "lbp_chi2",
    "freq_cue",
    "fft_hf_peak_ratio",
    "fft_hf_conc8",
    "seam_cue",
    "seam_hough_cue",
    "hog_asym",
)


def _image_document(
    bgr: np.ndarray, *, capture_layout: str = "single", capture_split_x: int | None = None
) -> ImageDocument:
    arr = np.ascontiguousarray(bgr)
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else int(arr.shape[2])
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    return ImageDocument(
        array=arr,
        width=w,
        height=h,
        channels=ch,
        image_format="png",
        sha256=sha,
        file_bytes=int(arr.nbytes),
        file_name="camera_frame.png",
        capture_layout=capture_layout,
        capture_split_x=capture_split_x,
    )


def _format_result(res: AlgorithmResult) -> str:
    verdict_label = strings.VERDICT_LABEL[res.verdict.value]
    lines = [
        strings.DEEPFAKE_PREVIEW_TITLE,
        strings.DEEPFAKE_PREVIEW_DISCLAIMER,
        f"Kết quả sàng lọc: {verdict_label}. {strings.DEEPFAKE_VERDICT_SCREEN}",
    ]
    evidence = res.evidence or {}
    if res.score is not None and not evidence.get("cue_unavailable"):
        lines.append(strings.DEEPFAKE_PREVIEW_SCORE.format(score=res.score))
    # Keep the reason visible even when the method produced a numeric score.
    # Otherwise a quality failure or missing ROI is hidden behind scalar metrics.
    lines.extend(dict.fromkeys(res.limitations[:2]))
    temporal_reason = evidence.get("temporal_reason")
    if temporal_reason and temporal_reason not in lines:
        lines.append(str(temporal_reason))
    has_temporal_face = bool(evidence.get("temporal_face_detected"))
    keys = (*_EVIDENCE_KEYS, "temporal_frame_count")
    if has_temporal_face:
        keys = (*keys, "temporal_static_score")
    for key in keys:
        if key in evidence:
            lines.append(f"{key}: {evidence[key]}")
    smooth = float(evidence.get("smooth_cue") or 0.0)
    lbp = float(evidence.get("lbp_cue") or 0.0)
    if (
        res.verdict.value == "no_indication"
        and evidence.get("face_detected")
        and not evidence.get("cue_unavailable")
        and smooth < 0.15
        and lbp < 0.15
    ):
        lines.append(strings.DEEPFAKE_PREVIEW_LIVE_TEXTURE)
    if (
        float(evidence.get("temporal_static_score") or 0.0) >= TEMPORAL_STATIC_MIN
        and has_temporal_face
    ):
        lines.append(strings.CAMERA_BURST_HINT)
    return "\n".join(lines)


def format_deepfake_preview(
    bgr: np.ndarray, *, capture_layout: str = "single", capture_split_x: int | None = None
) -> str:
    """Trả văn bản đa dòng: tiêu đề, disclaimer, verdict và evidence khi có."""
    res = DeepfakeCues().analyze(
        _image_document(bgr, capture_layout=capture_layout, capture_split_x=capture_split_x),
        load_thresholds(),
    )
    return _format_result(res)


def format_burst_deepfake_preview(frames: list[np.ndarray]) -> str:
    """Khung giữa kèm độ tĩnh của burst; độ tĩnh không cộng vào điểm Deepfake."""
    if not frames:
        return (
            f"{strings.DEEPFAKE_PREVIEW_TITLE}\n"
            f"{strings.DEEPFAKE_PREVIEW_DISCLAIMER}\n"
            f"{strings.DEEPFAKE_PREVIEW_NO_FRAMES}"
        )
    if len(frames) < 2:
        return format_deepfake_preview(frames[0])
    from imageguard.forensics.auxiliary.temporal_face_roi import apply_temporal_cue

    mid = frames[len(frames) // 2]
    base = DeepfakeCues().analyze(_image_document(mid), load_thresholds())
    return _format_result(apply_temporal_cue(base, frames))
