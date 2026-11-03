"""Enums dùng chung toàn hệ thống."""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """Kết luận pháp y ảnh. Có chủ đích KHÔNG dùng FORGED/FAKE:
    công cụ chỉ phát hiện *dấu hiệu*, không phán quyết sự thật."""

    SUSPICIOUS_INDICATION = "suspicious_indication"
    NO_INDICATION = "no_indication"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class MethodName(str, Enum):
    COPY_MOVE_KEYPOINT = "copy_move_keypoint"
    COPY_MOVE_BLOCK = "copy_move_block"
    ELA = "ela"
    EXIF_AUDIT = "exif_audit"
    NOISE_INCONSISTENCY = "noise_inconsistency"
    FREQUENCY_ANALYSIS = "frequency_analysis"
