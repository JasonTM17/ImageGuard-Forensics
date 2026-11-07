"""Course lock: a zero-radius disk is replaced by the next radius."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.enhance_ops import disk_kernel, wrong_kernel


def test_wrong_disk_radius_steps_up() -> None:
    params = {"radius": 0}
    kernel = wrong_kernel("disk", params)
    assert params["wrong_radius"] == 1
    np.testing.assert_array_equal(kernel, disk_kernel(1))
