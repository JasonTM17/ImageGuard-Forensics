"""Lock safe_open channel/dtype/dimension contract on edge images."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument


def test_safe_open_grayscale_png_returns_bgr_uint8_with_equal_channels(tmp_path: Path) -> None:
    gray = np.full((64, 64), 77, dtype=np.uint8)
    path = tmp_path / "gray.png"
    Image.fromarray(gray, mode="L").save(path, format="PNG")

    doc = safe_open(path, load_thresholds())
    assert isinstance(doc, ImageDocument)
    assert doc.array.shape == (64, 64, 3)
    assert doc.array.dtype == np.uint8
    assert doc.channels == 3
    assert np.all(doc.array == 77)


def test_safe_open_rgba_png_drops_alpha_and_keeps_bgr_order(tmp_path: Path) -> None:
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :] = [10, 20, 30, 128]  # R=10, G=20, B=30, A=128
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")

    doc = safe_open(path, load_thresholds())
    assert isinstance(doc, ImageDocument)
    assert doc.array.shape == (64, 64, 3)
    assert doc.array.dtype == np.uint8
    # OpenCV BGR order: [B=30, G=20, R=10]
    np.testing.assert_array_equal(doc.array[0, 0], np.array([30, 20, 10], dtype=np.uint8))


def test_safe_open_16bit_png_normalizes_to_uint8_bgr(tmp_path: Path) -> None:
    u16 = np.zeros((64, 64), dtype=np.uint16)
    u16[:32, :] = 65535
    u16[32:, :] = 0
    path = tmp_path / "bit16.png"
    ok, buf = cv2.imencode(".png", u16)
    assert ok
    path.write_bytes(buf.tobytes())

    doc = safe_open(path, load_thresholds())
    assert isinstance(doc, ImageDocument)
    assert doc.array.shape == (64, 64, 3)
    assert doc.array.dtype == np.uint8
    assert np.all(doc.array[:32, :] == 255)
    assert np.all(doc.array[32:, :] == 0)


def test_safe_open_palette_png_converts_to_bgr_uint8(tmp_path: Path) -> None:
    pal_img = Image.new("P", (64, 64))
    # Palette entry 0 -> RGB(12, 34, 56), entry 1 -> RGB(200, 100, 50)
    palette = [12, 34, 56, 200, 100, 50] + [0] * (768 - 6)
    pal_img.putpalette(palette)
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[32:, :] = 1
    pal_img.putdata(arr.ravel().tolist())
    path = tmp_path / "palette.png"
    pal_img.save(path, format="PNG")

    doc = safe_open(path, load_thresholds())
    assert isinstance(doc, ImageDocument)
    assert doc.array.shape == (64, 64, 3)
    assert doc.array.dtype == np.uint8
    np.testing.assert_array_equal(doc.array[0, 0], np.array([56, 34, 12], dtype=np.uint8))
    np.testing.assert_array_equal(doc.array[40, 0], np.array([50, 100, 200], dtype=np.uint8))


@pytest.mark.parametrize(("w", "h"), [(1, 1), (63, 64), (64, 63)])
def test_safe_open_rejects_too_small_images(tmp_path: Path, w: int, h: int) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    path = tmp_path / f"small_{w}x{h}.png"
    Image.fromarray(img).save(path, format="PNG")

    res = safe_open(path, load_thresholds())
    assert isinstance(res, ErrorResult)
    assert res.code == ErrorCode.IMAGE_TOO_SMALL


def test_safe_open_accepts_exact_min_side_64x64(tmp_path: Path) -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    path = tmp_path / "exact64.png"
    Image.fromarray(img).save(path, format="PNG")

    res = safe_open(path, load_thresholds())
    assert isinstance(res, ImageDocument)
    assert (res.width, res.height) == (64, 64)


def test_safe_open_rejects_max_pixels_and_corrupt_and_riff_non_webp(tmp_path: Path) -> None:
    # 1. Exceed max_pixels via config limit
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p100 = tmp_path / "100x100.png"
    Image.fromarray(img).save(p100, format="PNG")
    cfg = ThresholdSet(raw={"limits": {"max_pixels": {"value": 5000}}})
    r_px = safe_open(p100, cfg)
    assert isinstance(r_px, ErrorResult)
    assert r_px.code == ErrorCode.TOO_MANY_PIXELS

    # 2. Corrupt PNG header/IDAT -> DECODE_FAILED
    corrupt_png = tmp_path / "corrupt.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    r_corrupt = safe_open(corrupt_png, load_thresholds())
    assert isinstance(r_corrupt, ErrorResult)
    assert r_corrupt.code == ErrorCode.DECODE_FAILED

    # 3. Valid IHDR (128x128) with corrupt IDAT payload -> DECODE_FAILED during decode
    ihdr = struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0)
    ihdr_chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    bad_idat = struct.pack(">I", 4) + b"IDAT" + b"xxxx" + struct.pack(">I", zlib.crc32(b"IDATxxxx"))
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    bad_idat_png = tmp_path / "bad_idat.png"
    bad_idat_png.write_bytes(b"\x89PNG\r\n\x1a\n" + ihdr_chunk + bad_idat + iend)
    r_idat = safe_open(bad_idat_png, load_thresholds())
    assert isinstance(r_idat, ErrorResult)
    assert r_idat.code == ErrorCode.DECODE_FAILED

    # 4. RIFF header that is AVI (not WEBP) -> UNSUPPORTED_FORMAT
    avi = tmp_path / "fake.webp"
    avi.write_bytes(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 16)
    r_riff = safe_open(avi, load_thresholds())
    assert isinstance(r_riff, ErrorResult)
    assert r_riff.code == ErrorCode.UNSUPPORTED_FORMAT
