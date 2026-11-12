"""Edge-case contract tests for the Enhance derive engine and classical ops.

Scope: documented behaviour of ``imageguard.enhancement.derive`` and
``imageguard.evaluation.enhance_ops``. The NAFNet path needs model weights and
is intentionally out of scope. Every filesystem test uses ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.enhancement.derive import derive, write_sibling
from imageguard.evaluation.enhance_ops import (
    degrade,
    disk_kernel,
    gaussian_kernel,
    motion_kernel,
    restore,
)

_BGR = np.full((12, 14, 3), 90, dtype=np.uint8)
_UNSHARP = {"ksize": 5, "sigma": 1.0, "amount": 0.7}
_KERNEL = gaussian_kernel(3, 1.0).tolist()
_SMALL_NLM = {
    "h": 3.0,
    "hColor": 3.0,
    "templateWindowSize": 3,
    "searchWindowSize": 5,
}
_RESTORES = [
    ("unsharp", {"ksize": 5, "sigma": 1.0, "amount": 0.7}),
    ("clahe", {"clip_limit": 2.0, "tile_grid_size": [2, 2]}),
    ("nlm", dict(_SMALL_NLM)),
    (
        "richardson_lucy",
        {"iterations": 2, "kernel": _KERNEL, "kernel_role": "oracle"},
    ),
    ("wiener", {"nsr": 0.01, "kernel": _KERNEL, "kernel_role": "oracle"}),
]


def _png_bytes(image: np.ndarray) -> bytes:
    """Encode ``image`` as PNG bytes for source fixtures."""
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _source(
    tmp_path: Path, name: str = "frame.png", value: int = 90
) -> tuple[Path, np.ndarray, str]:
    """Write a PNG source and return (path, pixels, sha256 hex digest)."""
    image = np.full((18, 20, 3), value, dtype=np.uint8)
    path = tmp_path / name
    path.write_bytes(_png_bytes(image))
    return path, image, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("size", [1, 2, 3])
def test_restore_handles_one_pixel_and_tiny_bgr_images(size: int) -> None:
    """Every restoration operator keeps the shape of a 1x1..3x3 BGR image."""
    image = np.full((size, size, 3), 90, dtype=np.uint8)

    for kind, params in _RESTORES:
        output, metadata = restore(image, kind, params)
        assert output.shape == (size, size, 3)
        assert output.dtype == np.uint8
        assert metadata["kind"] == kind


@pytest.mark.parametrize("size", [1, 3])
def test_restore_keeps_tiny_grayscale_shape_for_channel_agnostic_ops(
    size: int,
) -> None:
    """unsharp, Richardson-Lucy and Wiener accept a 2-D image."""
    image = np.full((size, size), 90, dtype=np.uint8)
    cases = [
        ("unsharp", {"ksize": 3, "sigma": 1.0, "amount": 0.5}),
        (
            "richardson_lucy",
            {"iterations": 2, "kernel": _KERNEL, "kernel_role": "oracle"},
        ),
        ("wiener", {"nsr": 0.01, "kernel": _KERNEL, "kernel_role": "oracle"}),
    ]

    for kind, params in cases:
        output, _metadata = restore(image, kind, params)
        assert output.shape == (size, size)
        assert output.dtype == np.uint8


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("gaussian", {"ksize": 3, "sigma": 1.0}),
        ("disk", {"radius": 2}),
        ("motion", {"length": 3, "angle_deg": 30.0}),
        ("awgn", {"sigma": 2.0}),
        ("jpeg", {"quality": 50}),
    ],
)
def test_degrade_handles_one_pixel_image(
    kind: str, params: dict
) -> None:
    """Non-resizing degradations work on a 1x1 image."""
    image = np.full((1, 1, 3), 90, dtype=np.uint8)

    output, metadata = degrade(image, kind, params, rng=7)

    assert output.shape == image.shape
    assert metadata["kind"] == kind


def test_degrade_downsample_rejects_factor_that_empties_one_pixel() -> None:
    """A factor larger than the image cannot produce a non-zero dimension."""
    image = np.full((1, 1, 3), 90, dtype=np.uint8)

    with pytest.raises(ValueError, match="non-zero output dimension"):
        degrade(image, "downsample", {"factor": 2}, rng=None)


@pytest.mark.parametrize(("kind", "params"), _RESTORES[1:3])
@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4)], ids=["gray", "four-channel"])
def test_channel_operators_reject_non_three_channel_input(
    kind: str, params: dict, shape: tuple[int, ...]
) -> None:
    """CLAHE and NLM document three-channel BGR; other shapes raise ValueError."""
    image = np.full(shape, 90, dtype=np.uint8)

    with pytest.raises(ValueError, match="three-channel BGR"):
        restore(image, kind, params)


@pytest.mark.parametrize(
    ("kind", "params", "expected_shape"),
    [
        ("gaussian", {"ksize": 3, "sigma": 1.0}, (8, 8)),
        ("disk", {"radius": 2}, (8, 8)),
        ("motion", {"length": 3, "angle_deg": 30.0}, (8, 8)),
        ("awgn", {"sigma": 2.0}, (8, 8)),
        ("jpeg", {"quality": 50}, (8, 8)),
        ("downsample", {"factor": 2}, (4, 4)),
    ],
)
def test_degrade_accepts_grayscale(
    kind: str, params: dict, expected_shape: tuple[int, int]
) -> None:
    """Degradations are channel-agnostic and keep a 2-D image 2-D."""
    image = np.full((8, 8), 90, dtype=np.uint8)

    output, _metadata = degrade(image, kind, params, rng=3)

    assert output.shape == expected_shape


def test_unsharp_clips_out_of_range_uint16_and_float_to_uint8() -> None:
    """The documented uint8 output helper rounds and clips to [0, 255]."""
    floats = np.array([[-5.0, 100.0, 300.0]], dtype=np.float32)
    unsigned = np.array([[0, 300, 65535]], dtype=np.uint16)

    float_output, _ = restore(
        floats, "unsharp", {"ksize": 1, "sigma": 1.0, "amount": 0.0}
    )
    uint_output, _ = restore(
        unsigned, "unsharp", {"ksize": 1, "sigma": 1.0, "amount": 0.0}
    )

    assert float_output.dtype == np.uint8
    assert float_output.tolist() == [[0, 100, 255]]
    assert uint_output.dtype == np.uint8
    assert uint_output.tolist() == [[0, 255, 255]]


def test_kernel_restorers_convert_uint16_and_float_input_to_uint8() -> None:
    """Richardson-Lucy and Wiener compute in float and clip to uint8."""
    gray_uint16 = (np.arange(8 * 9, dtype=np.uint16) * 50).reshape(8, 9)
    bgr_float = (np.arange(8 * 9 * 3, dtype=np.float64) * 0.5).reshape(8, 9, 3)

    lucy, _ = restore(
        gray_uint16,
        "richardson_lucy",
        {"iterations": 2, "kernel": _KERNEL, "kernel_role": "oracle"},
    )
    wiener, _ = restore(
        bgr_float,
        "wiener",
        {"nsr": 0.01, "kernel": _KERNEL, "kernel_role": "oracle"},
    )

    assert lucy.shape == gray_uint16.shape
    assert lucy.dtype == np.uint8
    assert wiener.shape == bgr_float.shape
    assert wiener.dtype == np.uint8


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REPORT D-1: clahe and nlm pass non-uint8 BGR arrays to OpenCV and "
        "leak cv2.error instead of raising ValueError for an unsupported dtype"
    ),
)
@pytest.mark.parametrize(("kind", "params"), _RESTORES[1:3])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32], ids=["uint16", "float32"])
def test_channel_operators_reject_unsupported_dtype_with_value_error(
    kind: str, params: dict, dtype: type
) -> None:
    """Unsupported-but-well-shaped input should fail through ValueError."""
    image = np.full((6, 6, 3), 100, dtype=dtype)

    with pytest.raises(ValueError):
        restore(image, kind, params)


def test_kernels_larger_than_the_image_keep_the_image_shape() -> None:
    """A kernel may exceed the image; border handling is the operator's job."""
    tiny = np.full((9, 9, 3), 90, dtype=np.uint8)

    sharpened, _ = restore(
        tiny, "unsharp", {"ksize": 31, "sigma": 5.0, "amount": 0.5}
    )
    blurred, metadata = degrade(
        tiny, "gaussian", {"ksize": 31, "sigma": 5.0}, rng=None
    )
    disked, _ = degrade(tiny, "disk", {"radius": 12}, rng=None)

    assert sharpened.shape == tiny.shape
    assert blurred.shape == tiny.shape
    assert metadata["ksize"] == 31
    assert disked.shape == tiny.shape


def test_wiener_folds_kernel_larger_than_one_pixel_image() -> None:
    """The Wiener padding maps kernel taps modulo the image dimensions."""
    image = np.full((1, 1, 3), 90, dtype=np.uint8)

    output, _metadata = restore(
        image,
        "wiener",
        {"nsr": 0.01, "kernel": gaussian_kernel(5, 1.0).tolist(), "kernel_role": "oracle"},
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8


@pytest.mark.parametrize(
    ("kind", "params", "match"),
    [
        ("unsharp", {"ksize": -3, "sigma": 1.0, "amount": 0.5}, "at least 1"),
        ("unsharp", {"ksize": 3, "sigma": float("nan"), "amount": 0.5}, "finite"),
        ("unsharp", {"ksize": 3, "sigma": 1.0, "amount": -0.5}, "at least 0.0"),
        ("clahe", {"clip_limit": float("nan"), "tile_grid_size": [2, 2]}, "finite"),
        ("nlm", {**_SMALL_NLM, "h": -1.0}, "at least 0.0"),
        ("nlm", {**_SMALL_NLM, "templateWindowSize": 4}, "must be odd"),
        (
            "richardson_lucy",
            {"iterations": 0, "kernel": _KERNEL, "kernel_role": "oracle"},
            "at least 1",
        ),
        (
            "richardson_lucy",
            {"iterations": 2, "kernel": [[0.0, 0.0]], "kernel_role": "oracle"},
            "finite positive sum",
        ),
        (
            "richardson_lucy",
            {"iterations": 2, "kernel": [[float("nan")]], "kernel_role": "oracle"},
            "finite positive sum",
        ),
        (
            "wiener",
            {"nsr": -0.01, "kernel": _KERNEL, "kernel_role": "oracle"},
            "at least 0.0",
        ),
        (
            "wiener",
            {"nsr": 0.01, "kernel": _KERNEL, "kernel_role": "guess"},
            "kernel_role",
        ),
    ],
)
def test_restore_rejects_non_finite_or_negative_parameters(
    kind: str, params: dict, match: str
) -> None:
    """Bad parameters fail before any pixel is touched."""
    with pytest.raises(ValueError, match=match):
        restore(_BGR, kind, params)


@pytest.mark.parametrize(
    ("kind", "params", "match"),
    [
        ("gaussian", {"ksize": 4, "sigma": 1.0}, "must be odd"),
        ("gaussian", {"ksize": 3, "sigma": float("nan")}, "must be finite"),
        ("awgn", {"sigma": -1.0}, "at least 0.0"),
        ("jpeg", {"quality": 101}, "quality must be in"),
        ("downsample", {"factor": 0}, "at least 1"),
    ],
)
def test_degrade_rejects_non_finite_or_negative_parameters(
    kind: str, params: dict, match: str
) -> None:
    """Bad degradation parameters fail with ValueError."""
    with pytest.raises(ValueError, match=match):
        degrade(_BGR, kind, params, rng=1)


def test_integer_parameters_reject_bool_and_float() -> None:
    """Integer parameters are strict: no bools and no floats."""
    with pytest.raises(TypeError, match="integer"):
        restore(_BGR, "unsharp", {"ksize": True, "sigma": 1.0, "amount": 0.5})
    with pytest.raises(TypeError, match="integer"):
        degrade(_BGR, "downsample", {"factor": 2.0}, rng=None)


def test_kernel_helpers_reject_bad_dimensions_and_non_finite_values() -> None:
    """Kernel builders validate odd size, non-negative radius and finiteness."""
    with pytest.raises(ValueError, match="odd"):
        gaussian_kernel(4, 1.0)
    with pytest.raises(ValueError, match="finite"):
        gaussian_kernel(3, float("nan"))
    with pytest.raises(ValueError, match="at least 0"):
        disk_kernel(-1)
    with pytest.raises(ValueError, match="finite"):
        motion_kernel(3, float("inf"))
    with pytest.raises(TypeError, match="integer"):
        gaussian_kernel(True, 1.0)


def test_zero_radius_disk_is_a_normalized_identity_kernel() -> None:
    """radius=0 keeps a single unit tap that sums to one."""
    kernel = disk_kernel(0)

    assert kernel.shape == (1, 1)
    assert float(kernel.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros(5, dtype=np.uint8),
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
    ],
    ids=["empty", "one-dimensional", "four-dimensional"],
)
def test_restore_and_degrade_reject_non_image_arrays(image: np.ndarray) -> None:
    """Only non-empty 2-D or 3-D arrays are accepted."""
    with pytest.raises(ValueError, match="non-empty 2D or 3D"):
        restore(image, "unsharp", {"ksize": 3, "sigma": 1.0, "amount": 0.5})
    with pytest.raises(ValueError, match="non-empty 2D or 3D"):
        degrade(image, "gaussian", {"ksize": 3, "sigma": 1.0}, rng=None)


def test_unknown_operator_kinds_are_rejected() -> None:
    """Both entry points name the unsupported kind in the error."""
    with pytest.raises(ValueError, match="unsupported restoration kind: sharpen"):
        restore(_BGR, "sharpen", {})
    with pytest.raises(ValueError, match="unsupported degradation kind: blur"):
        degrade(_BGR, "blur", {}, rng=None)


def test_awgn_integer_seed_is_reproducible_and_recorded() -> None:
    """The same integer seed yields identical noise and is recorded."""
    image = np.arange(12 * 13 * 3, dtype=np.uint8).reshape(12, 13, 3)

    first, metadata = degrade(image, "awgn", {"sigma": 8.0}, rng=123)
    second, _ = degrade(image, "awgn", {"sigma": 8.0}, rng=123)

    assert metadata == {"kind": "awgn", "sigma": 8.0, "seed": 123}
    np.testing.assert_array_equal(first, second)


def test_awgn_generator_seed_is_reproducible_and_recorded() -> None:
    """A Generator exposes its entropy as the recorded seed."""
    image = np.arange(12 * 13 * 3, dtype=np.uint8).reshape(12, 13, 3)

    first, metadata = degrade(
        image, "awgn", {"sigma": 8.0}, rng=np.random.default_rng(7)
    )
    second, _ = degrade(
        image, "awgn", {"sigma": 8.0}, rng=np.random.default_rng(7)
    )
    different, _ = degrade(
        image, "awgn", {"sigma": 8.0}, rng=np.random.default_rng(8)
    )

    assert metadata["seed"] == 7
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_awgn_rejects_seed_that_disagrees_with_integer_rng() -> None:
    """Declaration and generator seed cannot disagree."""
    with pytest.raises(ValueError, match="must match the integer rng seed"):
        degrade(_BGR, "awgn", {"sigma": 4.0, "seed": 1}, rng=2)


def test_awgn_declared_seed_is_recorded_with_generator() -> None:
    """An explicit params seed overrides Generator entropy in metadata."""
    _, metadata = degrade(
        _BGR, "awgn", {"sigma": 4.0, "seed": 9}, rng=np.random.default_rng(3)
    )

    assert metadata["seed"] == 9


def test_restore_never_mutates_the_input_array() -> None:
    """Restoration reads a copy, even when the input is a strided view."""
    base = np.arange(16 * 18 * 3, dtype=np.uint8).reshape(16, 18, 3)
    view = base[:, ::2]
    before = view.copy()

    for kind, params in _RESTORES:
        output, _metadata = restore(view, kind, params)
        assert output is not view
        np.testing.assert_array_equal(view, before)

    np.testing.assert_array_equal(base[:, ::2], before)


def test_derive_never_mutates_input_and_is_repeatable() -> None:
    """derive copies the array and returns the same result for same params."""
    image = np.arange(20 * 22 * 3, dtype=np.uint8).reshape(20, 22, 3)
    before = image.copy()

    first, metadata = derive(image, "unsharp", dict(_UNSHARP))
    second, _ = derive(image, "unsharp", dict(_UNSHARP))

    assert metadata["kind"] == "unsharp"
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, before)


def test_derive_handles_one_pixel_and_grayscale_inputs() -> None:
    """derive delegates tiny and 2-D images without reshaping them."""
    color, _ = derive(np.full((1, 1, 3), 90, np.uint8), "unsharp", dict(_UNSHARP))
    gray, _ = derive(np.full((1, 1), 90, np.uint8), "unsharp", dict(_UNSHARP))

    assert color.shape == (1, 1, 3)
    assert color.dtype == np.uint8
    assert gray.shape == (1, 1)
    assert gray.dtype == np.uint8


def test_derive_rejects_unknown_kind_and_channel_mismatch() -> None:
    """derive surfaces operator validation instead of guessing."""
    with pytest.raises(ValueError, match="unsupported restoration kind"):
        derive(_BGR, "sharpen", {})
    with pytest.raises(ValueError, match="three-channel BGR"):
        derive(
            np.full((8, 8), 90, np.uint8),
            "clahe",
            {"clip_limit": 2.0, "tile_grid_size": [2, 2]},
        )


def test_write_sibling_names_sidecar_next_to_png_and_labels_it_derived(
    tmp_path: Path,
) -> None:
    """The sidecar replaces only the final suffix and says derived=True."""
    source, image, digest = _source(tmp_path)
    dest = tmp_path / "frame.derived.png"

    sidecar = write_sibling(
        source, dest, image, "unsharp", dict(_UNSHARP), digest
    )

    assert dest.exists()
    assert sidecar == tmp_path / "frame.derived.json"
    decoded = cv2.imdecode(
        np.frombuffer(dest.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert decoded.shape == image.shape
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["derived"] is True
    assert record["parent_sha256"] == digest
    assert record["derived_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()


def test_write_sibling_appends_json_suffix_when_destination_has_none(
    tmp_path: Path,
) -> None:
    """A suffix-less destination is written verbatim with a .json sibling."""
    source, image, digest = _source(tmp_path)
    dest = tmp_path / "frame_no_suffix"

    sidecar = write_sibling(
        source, dest, image, "unsharp", dict(_UNSHARP), digest
    )

    assert dest.exists()
    assert sidecar == tmp_path / "frame_no_suffix.json"


def test_write_sibling_refuses_parent_changed_after_digest_was_taken(
    tmp_path: Path,
) -> None:
    """A source edited after hashing is rejected before any output is written."""
    source, image, digest = _source(tmp_path)
    tampered_image = image.copy()
    tampered_image[:, :, 0] = 10
    tampered = _png_bytes(tampered_image)
    source.write_bytes(tampered)
    dest = tmp_path / "frame.derived.png"

    with pytest.raises(ValueError, match="parent_sha256 does not match"):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), digest)

    assert not dest.exists()
    assert not dest.with_suffix(".json").exists()
    assert source.read_bytes() == tampered


def test_write_sibling_refuses_existing_png_and_sidecar_without_overwrite(
    tmp_path: Path,
) -> None:
    """Collisions are reported and neither existing artifact is changed."""
    source, image, digest = _source(tmp_path)
    dest = tmp_path / "frame.derived.png"
    dest.write_bytes(b"existing png")
    dest.with_suffix(".json").write_text('{"existing": true}', encoding="utf-8")

    with pytest.raises(
        ValueError, match="destination already exists: frame.derived.png"
    ):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), digest)

    assert dest.read_bytes() == b"existing png"
    assert (
        dest.with_suffix(".json").read_text(encoding="utf-8")
        == '{"existing": true}'
    )


def test_write_sibling_overwrite_replaces_stale_png_and_sidecar(
    tmp_path: Path,
) -> None:
    """overwrite=True replaces both stale artifacts with one new pair."""
    source, image, digest = _source(tmp_path)
    dest = tmp_path / "frame.derived.png"
    dest.write_bytes(b"stale png")
    dest.with_suffix(".json").write_text('{"stale": true}', encoding="utf-8")

    sidecar = write_sibling(
        source, dest, image, "unsharp", dict(_UNSHARP), digest, overwrite=True
    )

    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert dest.read_bytes() != b"stale png"
    assert record["derived"] is True
    assert record["derived_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()


def test_write_sibling_never_mutates_the_written_array(tmp_path: Path) -> None:
    """Encoding reads the array without changing it, strided view included."""
    source, _image, digest = _source(tmp_path)
    base = np.arange(18 * 20 * 3, dtype=np.uint8).reshape(18, 20, 3)
    view = base[:, ::2]
    before = view.copy()

    write_sibling(source, tmp_path / "view.png", view, "unsharp", {}, digest)

    np.testing.assert_array_equal(view, before)
    np.testing.assert_array_equal(base[:, ::2], before)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "REPORT D-2: when dest ends in .json the PNG and its provenance "
        "sibling share one path; write_sibling overwrites the encoded PNG "
        "with the JSON record instead of raising ValueError"
    ),
)
def test_write_sibling_rejects_destination_whose_sidecar_is_itself(
    tmp_path: Path,
) -> None:
    """A destination that collides with its own sidecar must be refused."""
    source, image, digest = _source(tmp_path)
    dest = source.with_suffix(".json")
    original_source = source.read_bytes()

    with pytest.raises(ValueError):
        write_sibling(source, dest, image, "unsharp", dict(_UNSHARP), digest)

    assert source.read_bytes() == original_source
