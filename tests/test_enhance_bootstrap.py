"""Tests for reproducible percentile bootstrap intervals."""

from __future__ import annotations

import json

import numpy as np

from imageguard.evaluation.enhance_bootstrap import (
    bootstrap_mean,
    bootstrap_paired_diff,
    overlaps_zero,
)
from imageguard.evaluation.enhance_study import run_study
from imageguard.evaluation.synthforge import generate_set


def test_same_seed_reproduces_interval_bounds() -> None:
    values = [1.0, 2.0, 4.0, 8.0]

    first = bootstrap_mean(values, np.random.default_rng(17), n_boot=500)
    second = bootstrap_mean(values, np.random.default_rng(17), n_boot=500)

    assert first["low"] == second["low"]
    assert first["high"] == second["high"]


def test_single_finite_value_has_null_bounds() -> None:
    interval = bootstrap_mean([float("nan"), 3.0], np.random.default_rng(1))

    assert interval["n"] == 1
    assert interval["mean"] == 3.0
    assert interval["low"] is None
    assert interval["high"] is None


def test_identical_paired_values_have_zero_mean_difference() -> None:
    values = [0.5, 1.0, 2.0]

    interval = bootstrap_paired_diff(
        values,
        values,
        np.random.default_rng(4),
        n_boot=200,
    )

    assert interval["n_pairs"] == 3
    assert interval["mean"] == 0.0


def test_overlaps_zero_reports_interval_relation() -> None:
    assert overlaps_zero({"low": -1.0, "high": 1.0}) is True
    assert overlaps_zero({"low": 0.1, "high": 0.2}) is False
    assert overlaps_zero({"low": None, "high": None}) is None


def test_study_reports_arm_intervals_pairs_and_not_run_status(tmp_path) -> None:
    set_dir = tmp_path / "source"
    generate_set(1, seed=6, out_dir=set_dir, size=(64, 64), tier="easy")
    out_dir = tmp_path / "out"

    run_study(
        set_dir,
        out_dir,
        seed=23,
        degradations=["jpeg", "downsample"],
        restorers=["unsharp", "richardson_lucy"],
        methods=["noise"],
    )

    payload = json.loads((out_dir / "enhance_study.json").read_text(encoding="utf-8"))
    summaries = {
        summary["arm_id"]: summary
        for summary in payload["manipulated_arm_summaries"]
    }
    downsample = summaries["degradation:downsample"]
    assert downsample["bootstrap_intervals"]["psnr"]["n"] == 0
    assert downsample["bootstrap_intervals"]["psnr"]["omitted"] == 1
    assert downsample["bootstrap_intervals"]["keypoint_iou"]["n"] == 0
    assert downsample["bootstrap_intervals"]["block_iou"]["n"] == 0

    downsample_restore = summaries["restoration:downsample:unsharp:single"]
    psnr_diff = downsample_restore["paired_differences"]["psnr"]
    assert psnr_diff["n_pairs"] == 0
    assert psnr_diff["overlaps_zero"] is None

    not_run = summaries["restoration:jpeg:richardson_lucy:single"]
    assert not_run["status"] == "NOT_RUN"
    assert not_run["reason"] == "no generating blur PSF"

    matched_original_counts = payload["transition_counts"]["matched_original"]
    assert matched_original_counts["degradation:jpeg"]["noise"]["n"] == 1
    assert sum(
        matched_original_counts["degradation:jpeg"]["noise"]["counts"].values()
    ) == 1
    markdown = (out_dir / "enhance_study.md").read_text(encoding="utf-8")
    assert "A confidence interval which overlaps zero does not establish a change." in markdown
