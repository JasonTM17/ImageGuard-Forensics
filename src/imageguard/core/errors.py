"""Error taxonomy — mọi nhánh lỗi trả về mã có tên, không crash stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ErrorCode(str, Enum):
    PATH_INVALID = "PATH_INVALID"  # không tồn tại / không phải file
    FILE_UNREADABLE = "FILE_UNREADABLE"  # không đọc được bytes (permission, khóa)
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"  # magic bytes không khớp
    FILE_TOO_LARGE = "FILE_TOO_LARGE"  # vượt MAX_FILE_BYTES
    TOO_MANY_PIXELS = "TOO_MANY_PIXELS"  # header khai > MAX_PIXELS (bomb)
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"  # cạnh < MIN_SIDE
    DECODE_FAILED = "DECODE_FAILED"  # decoder không giải mã được (corrupt/truncated)
    INTERNAL = "INTERNAL"  # lỗi không phân loại được


@dataclass(slots=True)
class ErrorResult:
    """Kết quả lỗi trả về caller — UI/history vẫn lưu được, không raise."""

    code: ErrorCode
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


class ImageGuardError(Exception):
    """Exception nội bộ nếu cần raise sớm; boundary (service/UI) phải bắt
    và chuyển thành ErrorResult trước khi lên tới caller."""

    def __init__(self, code: ErrorCode, message: str, **detail: object) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail
