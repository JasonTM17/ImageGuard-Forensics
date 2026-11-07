"""Course lock: identical kernels have zero error and a one-row shift is fixed."""

from __future__ import annotations

import numpy as np

from imageguard.evaluation.blind_kernel import kernel_error


def test_kernel_error_identity_and_row_shift() -> None:
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    assert kernel_error(kernel, kernel) == 0.0
    assert kernel_error(kernel, np.roll(kernel, 1, axis=0)) == 0.6324555320336759
