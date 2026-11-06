"""Percentile bootstrap intervals; not BCa and not a p-value."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        raw_values = values
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [values]

    raw = np.asarray(raw_values, dtype=object).reshape(-1)
    converted = np.fromiter(
        (np.nan if value is None else float(value) for value in raw),
        dtype=np.float64,
        count=raw.size,
    )
    return converted


def _validate_n_boot(n_boot: int) -> int:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)):
        raise TypeError("n_boot must be an integer")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    return int(n_boot)


def bootstrap_mean(
    values: Iterable[float] | np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Return a percentile bootstrap interval for the finite sample mean."""
    n_boot = _validate_n_boot(n_boot)
    finite = _as_float_array(values)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)

    mean = float(np.mean(finite)) if n else None
    low: float | None = None
    high: float | None = None
    if n >= 2:
        indices = rng.integers(0, n, size=(n_boot, n))
        resample_means = np.mean(finite[indices], axis=1)
        low, high = (float(value) for value in np.percentile(resample_means, [2.5, 97.5]))

    return {
        "n": n,
        "mean": mean,
        "low": low,
        "high": high,
        "n_boot": n_boot,
        "method": "percentile",
    }


def bootstrap_paired_diff(
    enhanced: Iterable[float] | np.ndarray,
    degraded: Iterable[float] | np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Return a percentile bootstrap interval for paired enhanced-minus-degraded means."""
    enhanced_values = _as_float_array(enhanced)
    degraded_values = _as_float_array(degraded)
    pair_count = min(enhanced_values.size, degraded_values.size)
    enhanced_values = enhanced_values[:pair_count]
    degraded_values = degraded_values[:pair_count]

    finite_pair = np.isfinite(enhanced_values) & np.isfinite(degraded_values)
    differences = enhanced_values[finite_pair] - degraded_values[finite_pair]
    result = bootstrap_mean(differences, rng, n_boot=n_boot)
    result["n_pairs"] = int(differences.size)
    return result


def overlaps_zero(interval: dict[str, Any]) -> bool | None:
    """Return whether the interval includes zero, or ``None`` without both bounds."""
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return None
    return bool(low <= 0 <= high)
