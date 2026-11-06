"""Integrity lint — cấm chuỗi overclaim trong source code user-facing.

Tool này phát hiện DẤU HIỆU, không phán quyết — mọi string nói ngược lại
là bug khoa học, không phải vấn đề style.
"""

from __future__ import annotations

from pathlib import Path

BANNED = [
    "chắc chắn giả mạo",
    "xác suất giả mạo",
    "% tin cậy",
    "ảnh thật 100%",
    "definitely forged",
    "confirmed forgery",
    "is fake",
    "VERDICT_FORGED",
    "100% authentic",
    "đã bị giả mạo",
    "đã khôi phục",
    "ảnh rõ là ảnh thật",
    "enhancement xác nhận",
]


def test_no_overclaim_strings() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "imageguard"
    offenders: list[str] = []
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        for phrase in BANNED:
            if phrase.lower() in text:
                offenders.append(f"{f.name}: '{phrase}'")
    assert not offenders, f"overclaim strings found: {offenders}"


def test_verdict_enum_no_forge_labels() -> None:
    from imageguard.core.enums import Verdict

    names = {v.name for v in Verdict}
    assert "FORGED" not in names and "FAKE" not in names and "AUTHENTIC" not in names
