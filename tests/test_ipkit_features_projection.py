"""ipkit.features_projection — PCA / LDA (Ch.6)."""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.ipkit.features_projection import (
    lda_fit,
    lda_transform,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
)


def test_pca_recovers_known_axis() -> None:
    rng = np.random.default_rng(42)
    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    t = rng.normal(size=(80,))
    noise = rng.normal(scale=0.02, size=(80, 3))
    X = np.outer(t, direction) + noise
    _mean, comps, _, _ = pca_fit(X, 1)
    v = comps[0]
    cos = abs(float(np.dot(v, direction)))
    assert cos > 0.99


def test_pca_full_reconstruction_and_variance() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 5))
    mean, comps, ev, ratio = pca_fit(X, 5)
    Z = pca_transform(X, mean, comps)
    back = pca_inverse_transform(Z, mean, comps)
    np.testing.assert_allclose(back, X, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ratio.sum(), 1.0, rtol=1e-10, atol=1e-10)
    for j in range(5):
        col = Z[:, j]
        var = float(np.var(col, ddof=1))
        np.testing.assert_allclose(var, ev[j], rtol=1e-5, atol=1e-5)


def test_pca_matches_svd_oracle() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(size=(40, 8))
    mean, comps, ev, _ = pca_fit(X, 3)
    centered = X - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    oracle_comps = vt[:3]
    for i in range(3):
        assert abs(abs(float(np.dot(comps[i], oracle_comps[i]))) - 1.0) < 1e-10
    oracle_ev = (s[:3] ** 2) / 39.0
    np.testing.assert_allclose(ev, oracle_ev, rtol=1e-10, atol=1e-10)


def test_pca_bad_args() -> None:
    with pytest.raises(ValueError, match="n_components"):
        pca_fit(np.zeros((5, 3)), 10)
    with pytest.raises(TypeError):
        pca_fit(np.zeros((5, 3)), True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pca_fit(np.zeros((0, 3)), 1)


def test_lda_separates_gaussian_blobs() -> None:
    rng = np.random.default_rng(2026)
    n = 60
    c0 = rng.normal(loc=0.0, scale=0.4, size=(n, 4))
    c1 = rng.normal(loc=3.5, scale=0.4, size=(n, 4))
    X = np.vstack([c0, c1])
    y = np.array([0] * n + [1] * n)
    mean, comps, _ = lda_fit(X, y, 1)
    Z = lda_transform(X, mean, comps)
    z0 = Z[y == 0, 0]
    z1 = Z[y == 1, 0]
    between = abs(float(z0.mean() - z1.mean()))
    within = np.sqrt(float(np.var(z0, ddof=1) + np.var(z1, ddof=1)) / 2.0)
    assert between / max(within, 1e-9) > 4.0


def test_lda_deterministic_up_to_sign() -> None:
    X = np.vstack([np.zeros((8, 2)), np.ones((8, 2)) * 4])
    y = np.array([0] * 8 + [1] * 8)
    _, w1, _ = lda_fit(X, y, 1)
    _, w2, _ = lda_fit(X, y, 1)
    assert abs(abs(float(np.dot(w1[0], w2[0]))) - 1.0) < 1e-10


def test_lda_requires_two_classes() -> None:
    X = np.ones((4, 2))
    y = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match="2 classes"):
        lda_fit(X, y, 1)


def test_lda_n_components_bounds() -> None:
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="n_components"):
        lda_fit(X, y, 2)
