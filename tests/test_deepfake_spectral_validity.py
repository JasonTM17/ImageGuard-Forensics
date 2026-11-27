"""Numerical spectral controls, not an accuracy benchmark for Deepfakes.

White noise has no periodic grid. A sinusoid is a known periodic control,
but the same cue can occur in real fabrics, screens, and other textures.
"""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.forensics.auxiliary.deepfake_cues import _fft_highfreq_cue


@pytest.mark.parametrize("side", [32, 48, 64, 96, 128, 256])
def test_noise_cue_does_not_increase_for_small_rois(side: int) -> None:
    cues = []
    for seed in range(20):
        rng = np.random.default_rng(7000 + seed)
        noise = rng.normal(128.0, 10.0, (side, side)).astype(np.float32)
        cues.append(_fft_highfreq_cue(noise)[0])

    # The former fixed top-eight share gave ordinary 32px noise a cue
    # near 0.28, despite the same noise process being neutral at 128px.
    # These broad margins test that regression, not calibrated accuracy.
    assert float(np.mean(cues)) < 0.02
    assert max(cues) < 0.10


@pytest.mark.parametrize("side", [32, 48, 64, 96, 128, 256])
@pytest.mark.parametrize("period", [3.0, 4.0, 5.0])
def test_periodic_power_remains_visible_across_roi_areas(
    side: int, period: float,
) -> None:
    yy, xx = np.mgrid[:side, :side]
    for seed in range(8):
        rng = np.random.default_rng(19000 + seed)
        noise = rng.normal(128.0, 10.0, (side, side)).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        grid = 22.0 * np.cos(2.0 * np.pi * xx / period + phase)
        grid *= np.cos(2.0 * np.pi * yy / period + phase)
        noise_cue = _fft_highfreq_cue(noise)[0]
        grid_cue = _fft_highfreq_cue(noise + grid)[0]

        # Increasing crop area cannot hide a narrow periodic signal just
        # because fewer than 0.2% of bins contain the sinusoidal peaks.
        assert grid_cue > noise_cue + 0.4


@pytest.mark.parametrize("shape", [(48, 96), (63, 95), (97, 129)])
def test_odd_and_rectangular_rois_keep_noise_and_periodicity_separate(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(9821)
    noise = rng.normal(128.0, 10.0, shape).astype(np.float32)
    yy, xx = np.indices(shape)
    grid = 22.0 * np.cos(2.0 * np.pi * xx / 4.0) * np.cos(2.0 * np.pi * yy / 4.0)

    noise_result = _fft_highfreq_cue(noise)
    grid_result = _fft_highfreq_cue(noise + grid)
    assert np.all(np.isfinite(noise_result))
    assert np.all(np.isfinite(grid_result))
    assert noise_result[0] < 0.10
    assert grid_result[0] > noise_result[0] + 0.4


@pytest.mark.parametrize("value", [0.0, 128.0, 255.0])
def test_flat_roi_has_no_spectral_evidence(value: float) -> None:
    result = _fft_highfreq_cue(np.full((64, 96), value, dtype=np.float32))
    assert result[0] == 0.0
    assert np.all(np.isfinite(result))
