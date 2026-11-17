"""Course & contract tests cho toán tử siêu phân giải cổ điển (`super_res`)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from imageguard.enhancement.derive import derive
from imageguard.evaluation.enhance_ops import restore


def test_super_res_restores_same_shape_uint8_without_mutating_input() -> None:
    rng = np.random.default_rng(42)
    image = rng.integers(20, 220, (48, 64, 3), dtype=np.uint8)
    original = image.copy()

    out, meta = restore(image, "super_res", {"scale": 2, "iterations": 3, "amount": 0.8})

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert meta == {"kind": "super_res", "scale": 2, "iterations": 3, "amount": 0.8}
    np.testing.assert_array_equal(image, original)

    # Cũng chạy sạch qua entry point derive() cho cả grayscale lẫn BGR.
    out_gray, meta_gray = derive(image[:, :, 0], "super_res", {"scale": 2, "iterations": 2, "amount": 0.5})
    assert out_gray.shape == image[:, :, 0].shape
    assert out_gray.dtype == np.uint8
    assert meta_gray["kind"] == "super_res"


@pytest.mark.parametrize(
    ("params", "exc_type"),
    [
        ({"scale": 1, "iterations": 3, "amount": 0.8}, ValueError),
        ({"scale": 5, "iterations": 3, "amount": 0.8}, ValueError),
        ({"scale": True, "iterations": 3, "amount": 0.8}, TypeError),
        ({"scale": 2, "iterations": 0, "amount": 0.8}, ValueError),
        ({"scale": 2, "iterations": 20, "amount": 0.8}, ValueError),
        ({"scale": 2, "iterations": 3, "amount": -0.1}, ValueError),
    ],
)
def test_super_res_rejects_invalid_params(params: dict, exc_type: type[Exception]) -> None:
    image = np.full((24, 24, 3), 100, dtype=np.uint8)
    with pytest.raises(exc_type):
        restore(image, "super_res", params)


def test_enhance_screen_selects_super_res_operator(qtbot) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from imageguard.ui.screens.enhance import EnhanceScreen

    screen = EnhanceScreen()
    qtbot.addWidget(screen)
    idx = screen.combo_op.findData("super_res")
    assert idx >= 0
    screen.combo_op.setCurrentIndex(idx)
    params = screen._current_params()
    assert params == {"scale": 2, "iterations": 3, "amount": pytest.approx(0.8)}
