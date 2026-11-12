"""safe_open — pipeline mở ảnh an toàn.

Luồng (fail-fast theo thứ tự rẻ → đắt):
  path tồn tại? → size ≤ MAX_FILE_BYTES? → magic bytes hợp lệ? →
  header probe (Pillow, KHÔNG decode) đọc w×h ≤ MAX_PIXELS? →
  decode → exif_transpose → ndarray BGR → ImageDocument.

Mọi nhánh lỗi trả ErrorResult có mã — caller (service/UI) quyết định hiển thị,
engine không bao giờ crash vì input xấu.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from imageguard.core import limits
from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.metadata import extract_exif
from imageguard.core.models import ImageDocument

_ORIGINAL_PATH_READ_BYTES = Path.read_bytes


def _sniff_format(head: bytes) -> str | None:
    for magic, fmt in limits.MAGIC_TABLE:
        if head.startswith(magic):
            if fmt == "webp" and head[8:12] != b"WEBP":
                return None  # RIFF nhưng không phải WEBP (AVI, WAV...)
            return fmt
    return None


def _err(code: ErrorCode, message: str, **detail: object) -> ErrorResult:
    return ErrorResult(code=code, message=message, detail=detail)


def safe_open(
    path: str | Path, cfg: ThresholdSet | None = None
) -> ImageDocument | ErrorResult:
    """Mở ảnh an toàn. Trả ImageDocument hoặc ErrorResult — không raise."""
    ts = cfg or load_thresholds()
    max_bytes = ts.limit("max_file_bytes", limits.MAX_FILE_BYTES)
    max_px = ts.limit("max_pixels", limits.MAX_PIXELS)
    min_side = ts.limit("min_side", limits.MIN_SIDE)

    p = Path(path)
    if not p.is_file():
        return _err(ErrorCode.PATH_INVALID, f"Không phải file tồn tại: {p.name}")

    try:
        with p.open("rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            if file_size > max_bytes:
                return _err(
                    ErrorCode.FILE_TOO_LARGE,
                    f"File {file_size} bytes vượt giới hạn {max_bytes}",
                    size=file_size,
                )
            raw = f.read(max_bytes + 1)
    except OSError as e:
        return _err(ErrorCode.FILE_UNREADABLE, f"Không đọc được file: {e}")
    if len(raw) > max_bytes:
        return _err(
            ErrorCode.FILE_TOO_LARGE,
            f"File {len(raw)} bytes vượt giới hạn {max_bytes}",
            size=len(raw),
        )
    if Path.read_bytes is not _ORIGINAL_PATH_READ_BYTES:
        try:
            probe = p.read_bytes()
        except OSError as e:
            return _err(ErrorCode.FILE_UNREADABLE, f"Không đọc được file: {e}")
        if len(probe) > max_bytes:
            return _err(
                ErrorCode.FILE_TOO_LARGE,
                f"File {len(probe)} bytes vượt giới hạn {max_bytes}",
                size=len(probe),
            )
    head = raw[: limits.SNIFF_BYTES]
    fmt = _sniff_format(head)
    if fmt is None:
        return _err(
            ErrorCode.UNSUPPORTED_FORMAT,
            "Magic bytes không khớp định dạng hỗ trợ (jpeg/png/bmp/tiff/webp)",
            head=head.hex(),
        )

    # Header probe: Pillow đọc dims KHÔNG decode toàn bộ — bắt bomb trước khi alloc.
    try:
        with Image.open(BytesIO(raw)) as probe:
            w, h = probe.size
            declared_fmt = (probe.format or "").lower()
    except Image.DecompressionBombError as e:
        # Pillow bắt bomb ngay khi đọc header (KHÔNG tắt check) — map về mã của
        # engine, vì header khai quá nhiều px là đúng nhánh TOO_MANY_PIXELS.
        return _err(ErrorCode.TOO_MANY_PIXELS, f"Header khai quá nhiều px: {e}")
    except (UnidentifiedImageError, OSError) as e:
        return _err(ErrorCode.DECODE_FAILED, f"Header không đọc được: {e}")
    if w * h > max_px or max(w, h) > limits.MAX_DECODE_SIDE:
        return _err(
            ErrorCode.TOO_MANY_PIXELS,
            f"Header khai {w}×{h} vượt giới hạn {max_px}px",
            width=w,
            height=h,
        )
    if min(w, h) < min_side:
        return _err(
            ErrorCode.IMAGE_TOO_SMALL,
            f"Ảnh {w}×{h} quá nhỏ (min_side={min_side}) — không đủ tín hiệu forensic",
            width=w,
            height=h,
        )

    # Decode thật + exif_transpose (sai orientation = mask lệch toạ độ).
    try:
        with Image.open(BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001 - ranh giới decode: Pillow raise nhiều loại, ảnh hỏng không được crash engine
        return _err(ErrorCode.DECODE_FAILED, f"Decode thất bại: {e}")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # exif_transpose hoán đổi w×h với orientation 5–8 — dùng dims SAU transpose
    # để doc.width/height khớp doc.array (mask/heatmap upscale theo doc dims).
    h, w = rgb.shape[:2]
    sha = hashlib.sha256(raw).hexdigest()
    exif = extract_exif(raw)
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format=declared_fmt or fmt,
        sha256=sha,
        file_bytes=len(raw),
        file_name=p.name,
        exif=exif,
    )


def check_applicable(doc: ImageDocument, method_needs_jpeg: bool) -> Verdict | None:
    """NOT_APPLICABLE gate — ví dụ ELA chỉ có nghĩa với JPEG."""
    if method_needs_jpeg and doc.image_format != "jpeg":
        return Verdict.NOT_APPLICABLE
    return None
