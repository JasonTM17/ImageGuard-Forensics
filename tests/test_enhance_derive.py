"""Unit tests for the Qt-free Enhance engine (`imageguard.enhancement.derive`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.enhancement.derive import derive, write_sibling

_PARAMS = {"ksize": 5, "sigma": 1.0, "amount": 0.7}


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_derive_applies_operator_and_does_not_mutate_input() -> None:
    image = np.arange(20 * 22 * 3, dtype=np.uint8).reshape(20, 22, 3)
    original = image.copy()

    out, meta = derive(image, "unsharp", dict(_PARAMS))

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert meta["kind"] == "unsharp"
    np.testing.assert_array_equal(image, original)


def test_write_sibling_records_provenance_without_paths(tmp_path: Path) -> None:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    source.write_bytes(_png_bytes(image))
    parent_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    out, _meta = derive(image, "unsharp", dict(_PARAMS))
    dest = tmp_path / "goc_enhanced.png"
    json_path = write_sibling(
        source, dest, out, "unsharp", dict(_PARAMS), parent_sha)

    assert json_path == dest.with_suffix(".json")
    text = json_path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert set(record) == {
        "derived", "parent_sha256", "derived_sha256",
        "method", "params", "model_sha256", "created_utc",
    }
    assert record["derived"] is True
    assert record["parent_sha256"] == parent_sha
    assert record["model_sha256"] is None
    assert record["method"] == "unsharp"
    assert record["params"]["ksize"] == 5
    assert record["derived_sha256"] == hashlib.sha256(
        dest.read_bytes()).hexdigest()

    # Không đường dẫn tuyệt đối / ổ đĩa trong provenance.
    assert str(tmp_path) not in text
    assert str(tmp_path.resolve()) not in text
    assert "parent_name" not in record


def test_write_sibling_records_optional_model_hash(tmp_path: Path) -> None:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    source.write_bytes(_png_bytes(image))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model_digest = "a" * 64

    out, _meta = derive(image, "unsharp", dict(_PARAMS))
    sidecar = write_sibling(
        source,
        tmp_path / "goc_enhanced.png",
        out,
        "nafnet",
        {},
        digest,
        model_sha256=model_digest,
    )

    assert json.loads(sidecar.read_text(encoding="utf-8"))["model_sha256"] == model_digest


def test_write_sibling_refuses_source_and_existing_destination(
    tmp_path: Path,
) -> None:
    image = np.full((24, 26, 3), 90, dtype=np.uint8)
    source = tmp_path / "goc.png"
    source.write_bytes(_png_bytes(image))
    parent_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError):
        write_sibling(source, source, image, "unsharp", dict(_PARAMS), parent_sha)
    assert source.read_bytes() == _png_bytes(image)  # file gốc nguyên vẹn

    dest = tmp_path / "goc_enhanced.png"
    write_sibling(source, dest, image, "unsharp", dict(_PARAMS), parent_sha)
    with pytest.raises(ValueError):
        write_sibling(source, dest, image, "unsharp", dict(_PARAMS), parent_sha)
    before = dest.read_bytes()
    write_sibling(
        source, dest, image, "unsharp", dict(_PARAMS), parent_sha, overwrite=True
    )
    assert dest.read_bytes() == before

    with pytest.raises(ValueError):
        write_sibling(
            source, source, image, "unsharp", dict(_PARAMS), parent_sha, overwrite=True
        )
    assert source.read_bytes() == _png_bytes(image)

    sidecar_only = tmp_path / "anh.png"
    sidecar_only.with_suffix(".json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        write_sibling(
            source, sidecar_only, image, "unsharp", dict(_PARAMS), parent_sha
        )
    assert not sidecar_only.exists()
    assert sidecar_only.with_suffix(".json").read_text(encoding="utf-8") == "{}"
