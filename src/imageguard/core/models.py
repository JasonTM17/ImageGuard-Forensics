"""Domain models — dataclasses thuần, serialize-friendly, Qt-free.

Quy ước: mảng pixel là np.ndarray; `array` trong ImageDocument là BGR uint8
(chuẩn OpenCV). Mask là single-channel uint8 (0/255) hoặc float32 heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imageguard.core.enums import MethodName, Verdict


@dataclass(slots=True)
class ImageDocument:
    """Ảnh đã qua safe_open — pixel + provenance tối thiểu cho mọi phân tích."""

    array: np.ndarray  # BGR uint8, đã exif_transpose
    width: int
    height: int
    channels: int
    image_format: str  # jpeg | png | bmp | tiff | webp
    sha256: str  # hash của file bytes gốc — định danh ảnh trong history/report
    file_bytes: int
    file_name: str  # chỉ basename — tuyệt đối không lưu absolute path
    exif: dict[str, Any] = field(default_factory=dict)
    resolution_reduced: bool = False  # True nếu working copy bị downscale
    scale_factor: float = 1.0  # work_size = original * scale_factor


@dataclass(slots=True)
class AlgorithmResult:
    """Kết quả một phương pháp forensic. `score_definition` BẮT BUỘC khi có
    score — cấm số mơ hồ kiểu "confidence 73%"."""

    method: MethodName
    verdict: Verdict
    score: float | None = None
    score_definition: str = ""  # ví dụ: "số cặp keypoint match qua RANSAC affine"

    def __post_init__(self) -> None:
        # Fail-fast: score mà thiếu definition = con số mơ hồ trong report.
        if self.score is not None and not self.score_definition:
            raise ValueError(
                f"{self.method}: score={self.score} thiếu score_definition"
            )
    mask: np.ndarray | None = None  # uint8 H×W, vùng nghi vấn = 255
    heatmap: np.ndarray | None = None  # float32 H×W [0,1], mật độ nghi vấn
    evidence: dict[str, Any] = field(default_factory=dict)  # số liệu thô, giải thích được
    limitations: list[str] = field(default_factory=list)  # cảnh báo áp dụng cho kết quả này
    params_hash: str = ""  # sha256[:16] của snapshot tham số đã dùng
    duration_ms: int = 0


@dataclass(slots=True)
class ForensicReport:
    """Hồ sơ phân tích tổng hợp — đơn vị lưu history / xuất báo cáo."""

    report_id: str  # uuid4
    image_sha256: str
    image_name: str  # basename
    image_width: int
    image_height: int
    created_at: str  # ISO-8601 UTC
    tool_version: str
    env: dict[str, str] = field(default_factory=dict)  # {python, opencv, numpy}
    thresholds_hash: str = ""
    methods: list[AlgorithmResult] = field(default_factory=list)
    overall_verdict: Verdict = Verdict.INCONCLUSIVE
    limitations: list[str] = field(default_factory=list)
    disclaimer: str = (
        "Công cụ chỉ phát hiện dấu hiệu nghi vấn từ phân tích tự động — "
        "không khẳng định ảnh bị can thiệp, cũng không xác thực nguyên bản."
    )
