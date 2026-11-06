import numpy as np
import pytest

from imageguard.evaluation.enhance_metrics import align_mask, iou, psnr, ssim
from imageguard.evaluation.enhance_ops import (
    degrade,
    disk_kernel,
    gaussian_kernel,
    motion_kernel,
    restore,
    wrong_kernel,
)


@pytest.mark.parametrize(
    "kernel",
    [
        gaussian_kernel(7, 1.4),
        disk_kernel(4),
        motion_kernel(9, 32.0),
    ],
)
def test_blur_kernels_are_normalized(kernel):
    assert float(kernel.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kind", "params", "delta_key"),
    [
        ("gaussian", {"ksize": 7, "sigma": 1.4}, "wrong_sigma"),
        ("disk", {"radius": 3}, "wrong_radius"),
        (
            "motion",
            {"length": 9, "angle_deg": 20.0},
            "wrong_angle_deg",
        ),
    ],
)
def test_wrong_kernel_is_distinct_and_records_parameter(kind, params, delta_key):
    image = np.full((24, 28, 3), 80, dtype=np.uint8)
    _, metadata = degrade(image, kind, params, rng=None)
    actual = np.asarray(metadata["kernel"], dtype=np.float32)
    changed = dict(metadata)

    candidate = wrong_kernel(kind, changed)

    assert not np.array_equal(candidate, actual)
    assert delta_key in changed


def test_degrade_downsample_changes_shape_by_integer_factor_and_preserves_input():
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    original = image.copy()

    output, metadata = degrade(image, "downsample", {"factor": 3}, rng=None)

    assert output.shape == (6, 10, 3)
    assert metadata == {"kind": "downsample", "factor": 3}
    np.testing.assert_array_equal(image, original)


def test_awgn_records_seed_and_does_not_mutate_input():
    image = np.full((12, 13, 3), 100, dtype=np.uint8)
    original = image.copy()

    first, metadata = degrade(image, "awgn", {"sigma": 8.0}, rng=123)
    second, _ = degrade(image, "awgn", {"sigma": 8.0}, rng=123)

    assert metadata == {"kind": "awgn", "sigma": 8.0, "seed": 123}
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, original)


def test_restoration_operators_record_parameters_and_return_uint8_images():
    image = np.full((20, 22, 3), 80, dtype=np.uint8)
    kernel = gaussian_kernel(3, 1.0).tolist()
    cases = [
        ("unsharp", {"ksize": 3, "sigma": 1.0, "amount": 0.7}),
        ("clahe", {"clip_limit": 2.0, "tile_grid_size": [4, 4]}),
        (
            "nlm",
            {
                "h": 3.0,
                "hColor": 3.0,
                "templateWindowSize": 7,
                "searchWindowSize": 21,
            },
        ),
        (
            "richardson_lucy",
            {
                "iterations": 2,
                "kernel": kernel,
                "kernel_role": "oracle",
            },
        ),
        (
            "wiener",
            {"nsr": 0.01, "kernel": kernel, "kernel_role": "oracle"},
        ),
    ]
    for kind, params in cases:
        restored, metadata = restore(image, kind, params)
        assert restored.shape == image.shape
        assert restored.dtype == np.uint8
        assert metadata["kind"] == kind


def test_psnr_constant_one_level_shift_matches_hand_calculation():
    reference = np.full((12, 13, 3), 100, dtype=np.uint8)
    prediction = reference + np.uint8(1)

    assert psnr(prediction, reference) == pytest.approx(
        10 * np.log10(255**2 / 1)
    )


def test_psnr_identical_images_is_undefined():
    image = np.full((12, 13, 3), 40, dtype=np.uint8)

    assert psnr(image, image.copy()) is None


def test_ssim_identical_images_is_one():
    image = np.arange(12 * 13 * 3, dtype=np.uint8).reshape(12, 13, 3)

    assert ssim(image, image.copy()) == pytest.approx(1.0)


def test_iou_returns_none_for_shape_mismatch():
    assert iou(np.zeros((5, 6), dtype=np.uint8), np.zeros((6, 5), dtype=np.uint8)) is None


def test_iou_empty_prediction_on_nonempty_mask_is_zero():
    prediction = np.zeros((5, 6), dtype=np.uint8)
    ground_truth = np.zeros_like(prediction)
    ground_truth[2, 3] = 255

    assert iou(prediction, ground_truth) == 0.0


def test_iou_of_two_empty_masks_is_undefined():
    empty = np.zeros((5, 6), dtype=np.uint8)

    assert iou(empty, empty) is None


def test_align_mask_uses_nearest_neighbor_target_coordinates():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    aligned = align_mask(mask, (4, 4, 3))

    assert aligned.shape == (4, 4)
    assert set(np.unique(aligned)) == {0, 255}
