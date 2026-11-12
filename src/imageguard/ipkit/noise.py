"""Synthetic noise operators (NumPy only)."""

from __future__ import annotations

import numpy as np

from imageguard.ipkit._util import apply_per_channel, require_uint8_image


def _check_amount(amount: float) -> float:
    a = float(amount)
    if not np.isfinite(a) or a < 0.0 or a > 1.0:
        raise ValueError("amount must be in [0, 1]")
    return a


def _check_sigma(sigma: float) -> float:
    s = float(sigma)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("sigma must be a finite non-negative number")
    return s


def _check_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return int(seed)


def salt_pepper(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Salt-and-pepper noise: ``amount`` fraction of pixels set to 0 or 255."""
    source = require_uint8_image(img)
    prob = _check_amount(amount)
    rng = np.random.default_rng(_check_seed(seed))
    out = source.copy()

    def _one(channel: np.ndarray) -> np.ndarray:
        ch = channel.copy()
        flat = ch.reshape(-1)
        n_flip = round(prob * flat.size)
        if n_flip == 0:
            return ch
        idx = rng.choice(flat.size, size=n_flip, replace=False)
        salt = rng.random(n_flip) >= 0.5
        flat[idx[salt]] = 255
        flat[idx[~salt]] = 0
        return ch

    return apply_per_channel(out, _one) if source.ndim == 3 else _one(out)


def speckle(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Multiplicative speckle: ``I * (1 + sigma * N(0,1))``, clipped to uint8."""
    source = require_uint8_image(img)
    s = _check_sigma(sigma)
    rng = np.random.default_rng(_check_seed(seed))

    def _one(channel: np.ndarray) -> np.ndarray:
        base = channel.astype(np.float64)
        noise = 1.0 + s * rng.standard_normal(base.shape)
        return np.clip(np.rint(base * noise), 0, 255).astype(np.uint8)

    return apply_per_channel(source, _one) if source.ndim == 3 else _one(source)


def poisson(img: np.ndarray, seed: int) -> np.ndarray:
    """Shot noise: independent Poisson draws per pixel (lambda = intensity)."""
    source = require_uint8_image(img)
    rng = np.random.default_rng(_check_seed(seed))

    def _one(channel: np.ndarray) -> np.ndarray:
        lam = channel.astype(np.float64)
        noisy = rng.poisson(lam)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return apply_per_channel(source, _one) if source.ndim == 3 else _one(source)
