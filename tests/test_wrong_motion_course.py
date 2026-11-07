"""Course lock: a vertical motion kernel is replaced by a 105 degree kernel."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import motion_kernel, wrong_kernel


def test_vertical_motion_kernel_steps_to_105_degrees() -> None:
    params = {"length": 5, "angle_deg": 90.0}
    kernel = wrong_kernel("motion", params)
    assert params["wrong_angle_deg"] == 105.0
    np.testing.assert_array_equal(kernel, motion_kernel(5, 105.0))
