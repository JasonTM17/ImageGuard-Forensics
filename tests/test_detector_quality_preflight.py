"""Manifest preflight decode limits for detector-quality protocol."""

from __future__ import annotations

import struct
import zlib

import pytest
from PIL import Image

from scripts import evaluate_detector_quality as protocol


def _png_header_only(width: int, height: int) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = (
        struct.pack(">I", len(ihdr))
        + b"IHDR"
        + ihdr
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    )
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    return sig + ihdr_chunk + iend_chunk


def test_oversized_png_header_rejected_before_pixel_decode(tmp_path, monkeypatch):
    path = tmp_path / "bomb.png"
    path.write_bytes(_png_header_only(20_000, 20_000))
    called: list[bool] = []
    original_load = Image.Image.load

    def tracked_load(self: Image.Image, *args, **kwargs):
        called.append(True)
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "load", tracked_load)
    assert protocol._strict_pair(path, path, path) == "decode_limit_exceeded"
    assert not called


def test_enforce_decode_limits_rejects_claimed_dimensions(tmp_path):
    path = tmp_path / "bomb.png"
    path.write_bytes(_png_header_only(20_000, 20_000))
    with pytest.raises(ValueError, match="decode limits"):
        protocol._enforce_decode_limits(path)
