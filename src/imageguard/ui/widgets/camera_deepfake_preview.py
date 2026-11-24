"""Chuỗi xem trước dấu hiệu deepfake trên khung camera — không phụ thuộc Qt."""

from __future__ import annotations

import hashlib

import numpy as np

from imageguard.core.config import load_thresholds
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues
from imageguard.ui import strings

_EVIDENCE_KEYS = (
    "face_detected",
    "roi_source",
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


def _image_document(bgr: np.ndarray) -> ImageDocument:
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
    )


def _format_result(res) -> str:
    verdict_label = strings.VERDICT_LABEL[res.verdict.value]
    lines = [
        strings.DEEPFAKE_PREVIEW_TITLE,
        strings.DEEPFAKE_PREVIEW_DISCLAIMER,
        f"Kết quả sàng lọc: {verdict_label}. {strings.DEEPFAKE_VERDICT_SCREEN}",
    ]
    evidence = res.evidence or {}
    keys = list(_EVIDENCE_KEYS) + [
        "temporal_static_score",
        "temporal_frame_count",
    ]
    for key in keys:
        if key in evidence:
            lines.append(f"{key}: {evidence[key]}")
    smooth = float(evidence.get("smooth_cue") or 0.0)
    lbp = float(evidence.get("lbp_cue") or 0.0)
    if res.verdict.value == "no_indication" and smooth < 0.15 and lbp < 0.15:
        lines.append(strings.DEEPFAKE_PREVIEW_LIVE_TEXTURE)
    if float(evidence.get("temporal_static_score") or 0.0) >= 0.85:
        lines.append(strings.CAMERA_BURST_HINT)
    return "\n".join(lines)


def format_deepfake_preview(bgr: np.ndarray) -> str:
    """Trả văn bản đa dòng: tiêu đề, disclaimer, verdict và evidence khi có."""
    res = DeepfakeCues().analyze(_image_document(bgr), load_thresholds())
    return _format_result(res)


def format_burst_deepfake_preview(frames: list[np.ndarray]) -> str:
    """Khung giữa cộng điểm đứng yên. Không bao giờ ghi nhận diện deepfake."""
    if len(frames) < 2:
        return format_deepfake_preview(frames[0])
    from imageguard.forensics.auxiliary.temporal_face_roi import apply_temporal_cue

    mid = frames[len(frames) // 2]
    base = DeepfakeCues().analyze(_image_document(mid), load_thresholds())
    return _format_result(apply_temporal_cue(base, frames))