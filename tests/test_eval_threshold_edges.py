"""Edge cases for the threshold loader and the evaluation metrics.

Locks documented behavior only:

* ``load_thresholds`` / ``ThresholdSet.get`` → missing key, wrong node type,
  out-of-range value, unknown keys, hash stability and hash sensitivity.
* ``image_metrics`` / ``pixel_metrics`` / ``aggregate_pixel`` → empty masks,
  all-ones vs all-zeros, shape mismatch, single-sample splits, NaN masks and
  determinism.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from imageguard.core.config import (
    DEFAULT_CONFIG,
    Threshold,
    ThresholdSet,
    load_thresholds,
)
from imageguard.evaluation.metrics import (
    aggregate_pixel,
    image_metrics,
    pixel_metrics,
)

CONFIG_HASH_LEN = 16


def _write_yaml(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _mask(fill: int, shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    mask[:fill] = 255
    return mask


# --- threshold loader -------------------------------------------------------


def test_missing_key_raises_key_error() -> None:
    ts = ThresholdSet(raw={"keypoint": {}})

    with pytest.raises(KeyError) as excinfo:
        ts.get("keypoint", "absent_knob")
    assert "keypoint.absent_knob" in str(excinfo.value)

    with pytest.raises(KeyError):
        ts.get("absent_section", "absent_knob")


def test_missing_key_in_shipped_config_raises() -> None:
    ts = load_thresholds()

    with pytest.raises(KeyError):
        ts.get("keypoint", "definitely_not_a_threshold")


def test_non_mapping_threshold_node_is_rejected() -> None:
    ts = ThresholdSet(raw={"keypoint": {"ratio_rho": 0.8}})

    with pytest.raises(ValueError):
        ts.get("keypoint", "ratio_rho")


def test_threshold_node_without_value_is_rejected() -> None:
    ts = ThresholdSet(raw={"keypoint": {"ratio_rho": {"source": "s"}}})

    with pytest.raises(ValueError):
        ts.get("keypoint", "ratio_rho")


def test_non_mapping_yaml_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_thresholds(path)


def test_out_of_range_value_is_rejected_through_get() -> None:
    ts = ThresholdSet(
        raw={
            "keypoint": {
                "ratio_rho": {"value": 0.95, "source": "s", "range": [0.6, 0.9]}
            }
        }
    )

    with pytest.raises(ValueError):
        ts.get("keypoint", "ratio_rho")


def test_range_boundaries_are_inclusive() -> None:
    for value in (0.6, 0.9):
        Threshold(value=value, source="s", range=(0.6, 0.9)).validate()


def test_range_check_skips_non_numeric_values() -> None:
    Threshold(value="sift", source="s", range=(0.0, 1.0)).validate()


def test_unknown_keys_load_and_stay_lookupable(tmp_path: Path) -> None:
    payload = {
        "keypoint": {"ratio_rho": {"value": 0.8, "source": "s"}},
        "future_section": {"new_knob": {"value": 7, "source": "s"}},
    }
    ts = load_thresholds(_write_yaml(tmp_path, "extra.yaml", payload))

    assert ts.value("keypoint", "ratio_rho") == 0.8
    assert ts.value("future_section", "new_knob") == 7
    with pytest.raises(KeyError):
        ts.get("future_section", "not_in_file")


def test_hash_is_stable_across_repeated_loads() -> None:
    first = load_thresholds()
    second = load_thresholds()

    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == CONFIG_HASH_LEN


def test_hash_ignores_key_order(tmp_path: Path) -> None:
    first = _write_yaml(
        tmp_path,
        "first.yaml",
        {"a": {"y": {"value": 2, "source": "s"}},
         "b": {"x": {"value": 1, "source": "s"}}},
    )
    second = _write_yaml(
        tmp_path,
        "second.yaml",
        {"b": {"x": {"value": 1, "source": "s"}},
         "a": {"y": {"value": 2, "source": "s"}}},
    )

    assert load_thresholds(first).config_hash == load_thresholds(second).config_hash


def test_hash_ignores_version_field(tmp_path: Path) -> None:
    body = {"a": {"y": {"value": 2, "source": "s"}}}
    first = {"version": 1, **body}
    second = {"version": 99, **body}

    assert (
        load_thresholds(_write_yaml(tmp_path, "v1.yaml", first)).config_hash
        == load_thresholds(_write_yaml(tmp_path, "v99.yaml", second)).config_hash
    )


def test_changing_one_value_changes_the_hash(tmp_path: Path) -> None:
    baseline = load_thresholds()
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(data)
    changed["limits"]["min_side"]["value"] += 1

    altered = load_thresholds(_write_yaml(tmp_path, "altered.yaml", changed))

    assert altered.config_hash != baseline.config_hash
    assert (
        altered.value("limits", "min_side")
        == baseline.value("limits", "min_side") + 1
    )
    assert (
        altered.value("limits", "max_pixels")
        == baseline.value("limits", "max_pixels")
    )


def test_limit_uses_override_then_fallback() -> None:
    ts = ThresholdSet(raw={"limits": {"max_pixels": {"value": 12}}})

    assert ts.limit("max_pixels", fallback=99) == 12
    assert ts.limit("max_file_bytes", fallback=99) == 99


def test_get_exposes_value_source_unit_and_range() -> None:
    ts = load_thresholds()
    threshold = ts.get("keypoint", "ratio_rho")

    assert threshold.value == 0.8
    assert threshold.unit == "ratio"
    assert threshold.range == (0.6, 0.9)
    assert threshold.source


# --- image-level metrics ----------------------------------------------------


def test_image_metrics_single_true_sample() -> None:
    m = image_metrics([True], [True])

    assert m["n"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_image_metrics_single_false_positive_sample() -> None:
    m = image_metrics([True], [False])

    assert m["precision"] == 0.0
    assert m["f1"] == 0.0
    assert m["recall"] is None


def test_image_metrics_empty_split_has_no_numbers() -> None:
    m = image_metrics([], [])

    assert m["n"] == 0
    assert m["accuracy"] is None
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None


# --- pixel-level metrics ----------------------------------------------------


def test_pixel_metrics_both_masks_empty_is_r1() -> None:
    empty = np.zeros((4, 4), np.uint8)

    assert pixel_metrics(empty, empty) == {"skipped": "R1: gt empty"}
    assert pixel_metrics(None, empty) == {"skipped": "R1: gt empty"}


def test_pixel_metrics_zero_sized_gt_is_r1() -> None:
    empty = np.zeros((0, 0), np.uint8)

    assert pixel_metrics(empty, empty) == {"skipped": "R1: gt empty"}
    assert pixel_metrics(None, empty) == {"skipped": "R1: gt empty"}


def test_pixel_metrics_all_ones_against_all_ones_is_perfect() -> None:
    ones = np.ones((4, 4), np.uint8) * 255

    assert pixel_metrics(ones, ones) == {
        "skipped": None,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
        "tp": 16,
        "fp": 0,
        "fn": 0,
        "rule": None,
    }


def test_pixel_metrics_all_ones_prediction_against_empty_gt_is_r1() -> None:
    assert pixel_metrics(
        np.ones((4, 4), np.uint8) * 255, np.zeros((4, 4), np.uint8)
    ) == {"skipped": "R1: gt empty"}


def test_pixel_metrics_empty_prediction_against_all_ones_gt_is_r2() -> None:
    gt = np.ones((4, 4), np.uint8) * 255

    assert pixel_metrics(None, gt) == {
        "skipped": None,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 16,
        "rule": "R2",
    }


def test_pixel_metrics_all_ones_prediction_over_half_gt() -> None:
    gt = _mask(2)
    pred = np.ones((4, 4), np.uint8) * 255

    result = pixel_metrics(pred, gt)

    assert result["tp"] == 8
    assert result["fp"] == 8
    assert result["fn"] == 0
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == 1.0
    assert result["iou"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 * 0.5 / 1.5)


def test_pixel_metrics_rejects_3d_prediction_against_2d_gt() -> None:
    gt = _mask(1)

    with pytest.raises(ValueError):
        pixel_metrics(np.zeros((4, 4, 1), np.uint8), gt)


def test_pixel_metrics_treats_nan_as_background() -> None:
    nan_gt = np.full((2, 2), np.nan, dtype=np.float64)
    assert pixel_metrics(np.ones((2, 2), np.uint8), nan_gt) == {
        "skipped": "R1: gt empty"
    }

    gt = np.array([[0.0, 255.0], [0.0, 0.0]], dtype=np.float64)
    pred = np.array([[np.nan, 255.0], [np.nan, 0.0]], dtype=np.float64)
    result = pixel_metrics(pred, gt)

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["iou"] == 1.0


def test_pixel_metrics_are_deterministic() -> None:
    gt = _mask(2)
    pred = _mask(3)

    assert pixel_metrics(pred, gt) == pixel_metrics(pred, gt)


# --- aggregation ------------------------------------------------------------


def test_aggregate_single_sample_macro_equals_micro() -> None:
    gt = _mask(2)
    pred = _mask(2)

    agg = aggregate_pixel([pixel_metrics(pred, gt)])

    assert agg["n_images_scored"] == 1
    assert agg["macro"] == agg["micro"]
    assert agg["macro"]["iou"] == 1.0


def test_aggregate_only_skipped_rows_is_empty() -> None:
    empty = np.zeros((4, 4), np.uint8)

    agg = aggregate_pixel([pixel_metrics(empty, empty)])

    assert agg["n_images_scored"] == 0
    assert all(value is None for value in agg["macro"].values())
    assert all(value is None for value in agg["micro"].values())


def test_aggregate_is_order_independent() -> None:
    gt = _mask(2)
    results = [
        pixel_metrics(_mask(2), gt),
        pixel_metrics(None, gt),
        pixel_metrics(_mask(3), gt),
    ]

    assert aggregate_pixel(results) == aggregate_pixel(list(reversed(results)))
