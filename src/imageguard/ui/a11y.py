"""Gán accessibleName/Description cho widget — tên lấy từ ui/strings."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


def apply_a11y(widget: QWidget, name: str, description: str = "") -> None:
    """Đặt tên và (tuỳ chọn) mô tả cho trình đọc màn hình."""
    if not name.strip():
        raise ValueError("accessible name must be non-empty")
    widget.setAccessibleName(name)
    if description:
        widget.setAccessibleDescription(description)
