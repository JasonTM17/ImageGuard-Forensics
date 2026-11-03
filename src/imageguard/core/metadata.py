"""Đọc EXIF/metadata qua Pillow. Trả dict chuẩn hoá; missing tags ghi rõ —
thiếu metadata cũng là một quan sát (không phải lỗi)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

# Các tag quan trọng cho forensic — thiếu/bất thường đều đáng ghi nhận.
KEY_TAGS = ("Make", "Model", "Software", "DateTime", "DateTimeOriginal", "Orientation")


def extract_exif(path: Path | str) -> dict[str, Any]:
    """Đọc EXIF; trả {present: {...}, missing: [...], tag_count: int}.

    Không raise — ảnh không có EXIF vẫn là input hợp lệ.
    """
    result: dict[str, Any] = {"present": {}, "missing": [], "tag_count": 0}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            result["tag_count"] = len(exif)
            named = {TAGS.get(tag_id, str(tag_id)): str(val) for tag_id, val in exif.items()}
    except Exception:
        # Ảnh hợp lệ pixel nhưng EXIF hỏng → ghi nhận như missing toàn bộ.
        result["missing"] = list(KEY_TAGS)
        return result

    for tag in KEY_TAGS:
        if tag in named:
            result["present"][tag] = named[tag]
        else:
            result["missing"].append(tag)
    # IFD lồng (ExifOffset) — Pillow đã flatten một phần; DateTimeOriginal thường nằm đây.
    if "DateTimeOriginal" not in result["present"] and "ExifOffset" in named:
        try:
            with Image.open(path) as im:
                ifd = im.getexif().get_ifd(0x8769)
                for tag_id, val in ifd.items():
                    name = TAGS.get(tag_id, str(tag_id))
                    if name in KEY_TAGS and name not in result["present"]:
                        result["present"][name] = str(val)
                        result["missing"].remove(name)
        except Exception:
            pass
    return result
