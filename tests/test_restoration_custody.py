"""Tests cho custody, determinism, shape/dtype và edge cases của restoration/calibration."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.enhancement.derive import derive
from imageguard.evaluation.enhance_ops import gaussian_kernel, restore


def _sample_bgr(seed: int = 17, h: int = 48, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(15, 240, (h, w, 3), dtype=np.uint8)


def _sample_dual_bgr(left_closer_to_128: bool) -> np.ndarray:
    rng = np.random.default_rng(31)
    if left_closer_to_128:
        left = rng.integers(110, 146, (64, 80, 3), dtype=np.uint8)
        right = rng.integers(15, 70, (64, 80, 3), dtype=np.uint8)
    else:
        left = rng.integers(15, 70, (64, 80, 3), dtype=np.uint8)
        right = rng.integers(110, 146, (64, 80, 3), dtype=np.uint8)
    return np.hstack([left, right])


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("unsharp", {"ksize": 3, "sigma": 1.0, "amount": 0.8}),
        ("clahe", {"clip_limit": 2.0, "tile_grid_size": [4, 4]}),
        (
            "nlm",
            {
                "h": 3.0,
                "hColor": 3.0,
                "templateWindowSize": 5,
                "searchWindowSize": 11,
            },
        ),
        ("super_res", {"scale": 2, "iterations": 3, "amount": 0.8}),
        ("cam_calibrate", {"gamma": 0.85, "clip_limit": 0.04, "boost_k": 1.3}),
        ("freq_sharpen", {"d0": 35.0, "strength": 1.4, "median_k": 3}),
    ],
)
def test_restoration_ops_custody_determinism_and_finiteness(
    kind: str, params: dict
) -> None:
    image = _sample_bgr()
    original = image.copy()

    out1, meta1 = derive(image, kind, params)
    out2, meta2 = derive(image, kind, params)

    # Custody: input image must never be mutated or aliased.
    np.testing.assert_array_equal(image, original)
    assert out1 is not image
    assert out1.shape == image.shape
    assert out1.dtype == np.uint8
    assert np.all(np.isfinite(out1))

    # Determinism: identical inputs/params yield bit-exact outputs.
    np.testing.assert_array_equal(out1, out2)
    assert meta1 == meta2
    assert meta1["kind"] == kind


def test_unsharp_zero_amount_is_identity() -> None:
    image = _sample_bgr(seed=5, h=32, w=32)
    out, _ = restore(image, "unsharp", {"ksize": 3, "sigma": 1.0, "amount": 0.0})
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_super_res_scales_and_single_channel_3d_custody(scale: int) -> None:
    bgr = _sample_bgr(seed=7, h=24, w=32)
    orig_bgr = bgr.copy()
    out_bgr, meta = restore(bgr, "super_res", {"scale": scale, "iterations": 2, "amount": 0.6})
    assert out_bgr.shape == bgr.shape
    assert out_bgr.dtype == np.uint8
    assert meta["scale"] == scale
    np.testing.assert_array_equal(bgr, orig_bgr)

    # Single-channel 3D (H, W, 1) exercises cv2.resize ndim==2 squeeze guards.
    mono3d = bgr[:, :, :1].copy()
    orig_mono = mono3d.copy()
    out_mono, _ = restore(mono3d, "super_res", {"scale": scale, "iterations": 2, "amount": 0.6})
    assert out_mono.shape == mono3d.shape
    assert out_mono.dtype == np.uint8
    np.testing.assert_array_equal(mono3d, orig_mono)


@pytest.mark.parametrize("left_ref", [True, False])
def test_cam_calibrate_dual_matching_both_directions_and_custody(left_ref: bool) -> None:
    dual = _sample_dual_bgr(left_closer_to_128=left_ref)
    original = dual.copy()

    out1, meta1 = restore(
        dual, "cam_calibrate", {"gamma": 0.85, "clip_limit": 0.04, "boost_k": 1.2}
    )
    out2, _ = restore(
        dual, "cam_calibrate", {"gamma": 0.85, "clip_limit": 0.04, "boost_k": 1.2}
    )

    np.testing.assert_array_equal(dual, original)
    assert out1.shape == dual.shape
    assert out1.dtype == np.uint8
    assert meta1["dual_matched"] is True
    np.testing.assert_array_equal(out1, out2)

    # Histogram matching should pull the mismatched side's mean luminance closer to the reference side.
    mid = dual.shape[1] // 2
    in_diff = abs(float(dual[:, :mid].mean()) - float(dual[:, mid:].mean()))
    out_diff = abs(float(out1[:, :mid].mean()) - float(out1[:, mid:].mean()))
    assert out_diff < in_diff


def test_cam_calibrate_and_freq_sharpen_2d_grayscale_and_tiny_images() -> None:
    gray = _sample_bgr(seed=19, h=48, w=48)[:, :, 0].copy()
    orig = gray.copy()

    cc_out, cc_meta = restore(gray, "cam_calibrate", {})
    assert cc_out.shape == gray.shape and cc_out.dtype == np.uint8
    assert cc_meta["dual_matched"] is False
    np.testing.assert_array_equal(gray, orig)

    fs_out, fs_meta = restore(gray, "freq_sharpen", {"d0": 20.0, "strength": 1.0, "median_k": 3})
    assert fs_out.shape == gray.shape and fs_out.dtype == np.uint8
    assert fs_meta["kind"] == "freq_sharpen"
    np.testing.assert_array_equal(gray, orig)

    # Identity case: freq_sharpen with strength=0.0 and median_k=1 leaves low-freq / flat image intact.
    flat = np.full((16, 16), 120, dtype=np.uint8)
    fs_flat, _ = restore(flat, "freq_sharpen", {"d0": 20.0, "strength": 0.0, "median_k": 1})
    np.testing.assert_array_equal(fs_flat, flat)

    # Tiny 1x1 and 2x2 images must not crash.
    for hw in ((1, 1), (2, 2)):
        tiny = np.full(hw, 100, dtype=np.uint8)
        out_c, _ = restore(tiny, "cam_calibrate", {})
        out_f, _ = restore(tiny, "freq_sharpen", {})
        assert out_c.shape == hw and out_c.dtype == np.uint8
        assert out_f.shape == hw and out_f.dtype == np.uint8


def test_cam_calibrate_and_freq_sharpen_param_validation() -> None:
    img = _sample_bgr(h=32, w=32)
    with pytest.raises(ValueError, match="clip_limit"):
        restore(img, "cam_calibrate", {"clip_limit": 1.5})
    with pytest.raises(ValueError, match="median_k"):
        restore(img, "freq_sharpen", {"median_k": 2})


@pytest.mark.parametrize("kind", ["cam_calibrate", "freq_sharpen"])
@pytest.mark.parametrize("bad_dtype", [np.uint16, np.float32, np.int32])
def test_2d_calibration_and_freq_sharpen_reject_non_uint8_dtype(
    kind: str, bad_dtype: type
) -> None:
    arr = np.full((24, 24), 300, dtype=bad_dtype)
    with pytest.raises(ValueError, match="uint8"):
        restore(arr, kind, {})


def test_wiener_and_richardson_lucy_custody_and_determinism() -> None:
    img = _sample_bgr(seed=41, h=32, w=32)
    orig = img.copy()
    k = gaussian_kernel(3, 1.0).tolist()

    w1, _ = restore(img, "wiener", {"nsr": 0.02, "kernel": k, "kernel_role": "oracle"})
    w2, _ = restore(img, "wiener", {"nsr": 0.02, "kernel": k, "kernel_role": "oracle"})
    np.testing.assert_array_equal(img, orig)
    np.testing.assert_array_equal(w1, w2)
    assert w1.dtype == np.uint8 and w1.shape == img.shape

    rl1, _ = restore(
        img, "richardson_lucy", {"iterations": 2, "kernel": k, "kernel_role": "oracle"}
    )
    rl2, _ = restore(
        img, "richardson_lucy", {"iterations": 2, "kernel": k, "kernel_role": "oracle"}
    )
    np.testing.assert_array_equal(img, orig)
    np.testing.assert_array_equal(rl1, rl2)
    assert rl1.dtype == np.uint8 and rl1.shape == img.shape
