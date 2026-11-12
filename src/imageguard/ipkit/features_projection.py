"""
Giảm chiều dữ liệu: PCA và LDA (Chương 6 — Feature Extraction, HCMUTE dunghv).

PCA: không giám sát, phương sai lớn nhất trên không gian chiếu mới.
LDA: tối đa hóa tách lớp qua Sw, Sb và bài toán riêng inv(Sw) Sb (ridge khi Sw suy biến).
"""

from __future__ import annotations

import numpy as np

_LDA_RIDGE = 1e-6


def _as_2d_float(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    return arr


def _validate_n_components_pca(
    n_samples: int, n_features: int, n_components: int
) -> int:
    if isinstance(n_components, bool) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise TypeError("n_components must be an integer")
    k = int(n_components)
    upper = min(n_features, n_samples)
    if k < 1 or k > upper:
        raise ValueError(f"n_components must be in [1, {upper}]")
    return k


def pca_fit(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ước lượng PCA bằng SVD trên ma trận đã trừ trung bình (Ch.6 PCA, bước 1–4).

    Thay vì hiệp phương sai N×N (tốn khi N lớn), dùng SVD: X̃ = U Σ Vᵀ;
    vector riêng của hiệp phương sai mẫu nằm ở các hàng Vᵀ, phương sai giải thích ∝ σ²/(M−1).

    Parameters
    ----------
    X :
        Ma trận M×N (M mẫu, N đặc trưng).
    n_components :
        Số thành phần chính K (K ≤ min(N, M)).

    Returns
    -------
    mean :
        Trung bình mẫu shape (N,).
    components :
        K vector chính đơn vị shape (K, N).
    explained_variance :
        Phương sai giải thích theo từng thành phần shape (K,).
    explained_ratio :
        Tỉ lệ phương sai shape (K,), tổng ≤ 1.

    Example
    -------
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 4))
    >>> mean, comps, var, ratio = pca_fit(X, 2)
    >>> Z = pca_transform(X, mean, comps)
    >>> Z.shape
    (50, 2)
    """
    data = _as_2d_float(X)
    n_samples, n_features = data.shape
    k = _validate_n_components_pca(n_samples, n_features, n_components)

    # Trung bình mẫu μ và tâm hóa từng hàng (bước 1–2 slide)
    mean = data.mean(axis=0)
    centered = data - mean

    # SVD đầy đủ cột: tránh tạo ma trận hiệp phương sai N×N khi N >> M
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)

    # Phương sai mẫu không chệch: λ_i = σ_i² / (M−1)
    denom = max(n_samples - 1, 1)
    explained_variance = (singular**2) / denom
    total = float(explained_variance.sum())
    if total <= 0:
        explained_ratio = np.zeros(k, dtype=np.float64)
    else:
        explained_ratio = explained_variance / total

    # Lấy K hàng đầu của Vᵀ làm trục chính (đã chuẩn hóa độ dài 1)
    components = vt[:k].copy()
    return (
        mean,
        components,
        explained_variance[:k].copy(),
        explained_ratio[:k].copy(),
    )


def pca_transform(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """
    Chiếu dữ liệu lên không gian PCA: z = (x − μ) Wᵀ (Ch.6 PCA bước 5).

    Parameters
    ----------
    X :
        M×N hoặc bất kỳ tập điểm cùng số cột N.
    mean :
        μ shape (N,).
    components :
        W shape (K, N).

    Returns
    -------
    np.ndarray
        Ma trận M×K tọa độ trong không gian chính.

    Example
    -------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mean, W, _, _ = pca_fit(X, 1)
    >>> pca_transform(X, mean, W).shape
    (2, 1)
    """
    data = _as_2d_float(X)
    mu = np.asarray(mean, dtype=np.float64)
    w = np.asarray(components, dtype=np.float64)
    if mu.shape != (data.shape[1],):
        raise ValueError("mean shape must match number of features")
    if w.ndim != 2 or w.shape[1] != data.shape[1]:
        raise ValueError("components must have shape (k, n_features)")
    # Chiếu: mỗi hàng nhân với Wᵀ
    return (data - mu) @ w.T


def pca_inverse_transform(
    Z: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """
    Tái dựng x̂ ≈ μ + z W (giữ K thành phần chính).

    Parameters
    ----------
    Z :
        M×K tọa độ đã chiếu.
    mean, components :
        Như ``pca_fit``.

    Returns
    -------
    np.ndarray
        Ma trận M×N xấp xỉ trong không gian gốc.

    Example
    -------
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> mean, W, _, _ = pca_fit(X, 1)
    >>> recon = pca_inverse_transform(pca_transform(X, mean, W), mean, W)
    """
    z = _as_2d_float(Z, name="Z")
    mu = np.asarray(mean, dtype=np.float64)
    w = np.asarray(components, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != z.shape[1]:
        raise ValueError("components rows must match Z columns")
    if mu.shape != (w.shape[1],):
        raise ValueError("mean shape must match n_features")
    return z @ w + mu


def _validate_labels(y: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError("y must be 1D with one label per sample")
    return labels


def lda_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    ridge: float = _LDA_RIDGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Học LDA Fisher: tối đa hóa tỉ lệ phương sai liên / nội lớp (Ch.6 LDA 1/6–3/6).

    Sw = Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ; Sb = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ.
    Giải (Sw + λI)⁻¹ Sb w = λw với ridge λ nhỏ khi Sw suy biến (M << N).

    Parameters
    ----------
    X :
        M×N.
    y :
        Nhãn lớp cho từng hàng.
    n_components :
        Số chiều chiếu (≤ số lớp − 1).
    ridge :
        Hệ số λ cộng trên đường chéo Sw.

    Returns
    -------
    mean :
        Trung bình tổng μ.
    components :
        Các vector Fisher shape (K, N), hàng đơn vị.
    eigenvalues :
        Trị riêng tương ứng (giảm dần).

    Example
    -------
    >>> X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 3])
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> mean, W, ev = lda_fit(X, y, 1)
    """
    data = _as_2d_float(X)
    n_samples, n_features = data.shape
    labels = _validate_labels(y, n_samples)

    if isinstance(n_components, bool) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise TypeError("n_components must be an integer")
    k = int(n_components)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size
    if n_classes < 2:
        raise ValueError("LDA requires at least 2 classes")
    max_k = n_classes - 1
    if k < 1 or k > max_k:
        raise ValueError(f"n_components must be in [1, {max_k}]")

    lam = float(ridge)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("ridge must be a finite non-negative float")

    # Trung bình tổng và phân tán trong / giữa lớp
    mean = data.mean(axis=0)
    sw = np.zeros((n_features, n_features), dtype=np.float64)
    sb = np.zeros((n_features, n_features), dtype=np.float64)

    for c_idx in range(n_classes):
        mask = inverse == c_idx
        xc = data[mask]
        mu_c = xc.mean(axis=0)
        n_c = xc.shape[0]
        diff = xc - mu_c
        # Cộng dồn scatter nội lớp (không chia n_c để khớp dạng tổng slide)
        sw += diff.T @ diff
        between = (mu_c - mean).reshape(-1, 1)
        sb += n_c * (between @ between.T)

    # Ridge ổn định nghịch đảo Sw khi N >> M
    sw_reg = sw + lam * np.eye(n_features, dtype=np.float64)

    # Bài toán tổng quát: A = Sw⁻¹ Sb (không đối xứng)
    try:
        a_mat = np.linalg.solve(sw_reg, sb)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "within-class scatter matrix is singular even with ridge"
        ) from err

    eig_vals, eig_vecs = np.linalg.eig(a_mat)
    order = np.argsort(np.real(eig_vals))[::-1]
    eig_vals = np.real(eig_vals[order])
    eig_vecs = np.real(eig_vecs[:, order])

    # Chuẩn hóa hàng thành vector Fisher đơn vị
    raw_w = eig_vecs[:, :k].T
    norms = np.linalg.norm(raw_w, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    components = raw_w / norms

    return mean, components, eig_vals[:k].copy()


def lda_transform(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """
    Chiếu mẫu lên trục Fisher: y = (x − μ) Wᵀ.

    Parameters
    ----------
    X, mean, components :
        Giống đầu ra ``lda_fit``.

    Returns
    -------
    np.ndarray
        M×K.

    Example
    -------
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    >>> y = np.array([0, 0, 1])
    >>> mu, W, _ = lda_fit(X, y, 1)
    >>> lda_transform(X, mu, W).shape
    (3, 1)
    """
    return pca_transform(X, mean, components)
