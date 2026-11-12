"""Regression: write_sibling must verify parent_sha256 against source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.enhancement.derive import _sha256_file, write_sibling

_PARAMS = {"ksize": 5, "sigma": 1.0, "amount": 0.7}


def _write_source(tmp_path: Path) -> tuple[Path, np.ndarray, str]:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    source.write_bytes(encoded.tobytes())
    parent_sha = _sha256_file(source)
    return source, image, parent_sha


def test_write_sibling_rejects_wrong_parent_digest_without_creating_files(
    tmp_path: Path,
) -> None:
    source, image, _correct = _write_source(tmp_path)
    dest = tmp_path / "goc_enhanced.png"
    wrong = "0" * 64

    with pytest.raises(ValueError, match="parent_sha256 does not match"):
        write_sibling(source, dest, image, "unsharp", dict(_PARAMS), wrong)

    assert not dest.exists()
    assert not dest.with_suffix(".json").exists()


def test_write_sibling_rejects_wrong_digest_even_when_overwrite_would_apply(
    tmp_path: Path,
) -> None:
    source, image, correct = _write_source(tmp_path)
    dest = tmp_path / "goc_enhanced.png"
    write_sibling(source, dest, image, "unsharp", dict(_PARAMS), correct)
    before_png = dest.read_bytes()
    before_json = dest.with_suffix(".json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="parent_sha256 does not match"):
        write_sibling(
            source, dest, image, "unsharp", dict(_PARAMS), "f" * 64, overwrite=True
        )

    assert dest.read_bytes() == before_png
    assert dest.with_suffix(".json").read_text(encoding="utf-8") == before_json


def test_write_sibling_accepts_matching_parent_digest(tmp_path: Path) -> None:
    source, image, parent_sha = _write_source(tmp_path)
    dest = tmp_path / "goc_enhanced.png"

    sidecar = write_sibling(source, dest, image, "unsharp", dict(_PARAMS), parent_sha)

    assert dest.exists()
    record_parent = json.loads(sidecar.read_text(encoding="utf-8"))["parent_sha256"]
    assert record_parent == parent_sha
    assert record_parent == _sha256_file(source)
    assert record_parent == hashlib.sha256(source.read_bytes()).hexdigest()
