"""Focused checks for experimental blind-kernel estimation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.evaluation.blind_kernel import estimate_kernel, kernel_error
from imageguard.evaluation.enhance_ops import disk_kernel, restore
from imageguard.evaluation.enhance_study import DEFAULT_RESTORERS, run_study
from imageguard.evaluation.synthforge import generate_set


def _straight_edge_fixture() -> tuple[np.ndarray, np.ndarray]:
    sharp = np.full((128, 128, 3), 28, dtype=np.uint8)
    sharp[:, 64:, :] = 218
    truth = disk_kernel(4)
    blurred = cv2.filter2D(sharp, -1, truth)
    return blurred, truth


def test_disk_edge_estimate_has_finite_kernel_error_below_half() -> None:
    blurred, truth = _straight_edge_fixture()

    estimated, record = estimate_kernel(blurred)

    assert record["status"] == "RUN"
    assert estimated is not None
    assert np.all(np.isfinite(estimated))
    assert kernel_error(estimated, truth) < 0.5


def test_flat_image_is_not_run() -> None:
    estimated, record = estimate_kernel(np.full((64, 64, 3), 90, dtype=np.uint8))

    assert estimated is None
    assert record["status"] == "NOT_RUN"
    assert "edge" in record["reason"]


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("richardson_lucy", {"iterations": 2}),
        ("wiener", {"nsr": 0.01}),
    ],
)
@pytest.mark.parametrize("role", ["estimated", "oracle"])
def test_kernel_restorers_accept_estimated_and_oracle_roles(
    kind: str, params: dict, role: str
) -> None:
    image = np.full((24, 28, 3), 80, dtype=np.uint8)
    arguments = {
        **params,
        "kernel": disk_kernel(2),
        "kernel_role": role,
    }

    restored, metadata = restore(image, kind, arguments)

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    assert metadata["kernel_role"] == role


def test_kernel_restorers_reject_unknown_kernel_roles() -> None:
    with pytest.raises(ValueError, match="kernel_role"):
        restore(
            np.full((20, 20, 3), 80, dtype=np.uint8),
            "wiener",
            {"nsr": 0.01, "kernel": disk_kernel(2), "kernel_role": "other"},
        )


def test_awgn_blind_restoration_is_not_run_for_missing_blur_psf(
    tmp_path: Path,
) -> None:
    set_dir = tmp_path / "source"
    generate_set(1, seed=20261106, out_dir=set_dir, size=(96, 96), tier="easy")
    out_dir = tmp_path / "out"

    run_study(
        set_dir,
        out_dir,
        seed=20261106,
        degradations=["awgn"],
        restorers=["blind_lucy"],
        methods=["noise"],
    )

    payload = json.loads(
        (out_dir / "enhance_study.json").read_text(encoding="utf-8")
    )
    manipulated = next(
        sample for sample in payload["samples"]
        if sample["role"] == "manipulated_sample"
    )
    arm = next(
        candidate for candidate in manipulated["arms"]
        if candidate["arm"] == "restoration"
        and candidate["restorer"] == "blind_lucy"
    )
    assert arm["kernel_role"] == "estimated"
    assert arm["status"] == "NOT_RUN"
    assert arm["reason"] == "no generating blur PSF"


def test_optional_restorers_do_not_change_classical_defaults() -> None:
    assert DEFAULT_RESTORERS == (
        "unsharp",
        "clahe",
        "nlm",
        "richardson_lucy",
        "wiener",
    )


def test_estimator_module_has_no_fusion_or_ui_imports() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "imageguard"
        / "evaluation"
        / "blind_kernel.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any("fusion" in name or "imageguard.ui" in name for name in imported)
    assert "imageguard.fusion" not in source
    assert "imageguard.ui" not in source
