"""ipkit.classifiers — bộ phân lớp cơ bản (Chương 7: Pattern Recognition).

Các bài kiểm tra dùng dữ liệu nhỏ sinh ngẫu nhiên có seed cố định, không cần
mạng, không ghi file. Không có tuyên bố nào về độ chính xác trên dữ liệu giám
định ảnh: các bộ phân lớp này chỉ để minh họa thuật toán.
"""

from __future__ import annotations

import numpy as np
import pytest

from imageguard.ipkit.classifiers import (
    KNNClassifier,
    LinearSVM,
    MLPClassifier,
    OneVsRestSVM,
    accuracy_score,
    confusion_matrix,
    conv2d_valid,
    flatten,
    max_pool2d,
    minimum_distance_classifier,
    relu,
    softmax,
)

# ---------------------------------------------------------------------------
# Tiện ích sinh dữ liệu nhỏ (seed cố định, tái lập được)
# ---------------------------------------------------------------------------


def _make_binary_blobs(n: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Hai đám mây tách tuyến tính, nhãn {-1, +1}, tâm (-2,-2) và (2,2)."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal(-2.0, 0.3, (n, 2))
    c1 = rng.normal(2.0, 0.3, (n, 2))
    X = np.vstack([c0, c1])
    y = np.array([-1] * n + [1] * n)
    return X, y


def _make_multiclass_blobs(n: int = 20, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Ba đám mây tại ba đỉnh tam giác, mỗi lớp đều tách được kiểu one-vs-rest.

    Tâm (0,0), (6,0), (3,5) được chọn sao cho mỗi lớp đều nằm về một phía của
    một đường thẳng so với hai lớp còn lại (các đám thẳng hàng thì lớp giữa
    KHÔNG tách được kiểu one-vs-rest).
    """
    rng = np.random.default_rng(seed)
    c0 = rng.normal([0.0, 0.0], 0.3, (n, 2))
    c1 = rng.normal([6.0, 0.0], 0.3, (n, 2))
    c2 = rng.normal([3.0, 5.0], 0.3, (n, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


def _ref_corr2d(x: np.ndarray, k: np.ndarray, stride: int) -> np.ndarray:
    """Tương quan chéo "valid" tham chiếu bằng vòng lặp thuần NumPy (không scipy)."""
    H, W = x.shape
    kh, kw = k.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = x[i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[i, j] = np.sum(patch * k)
    return out


# ---------------------------------------------------------------------------
# minimum_distance_classifier
# ---------------------------------------------------------------------------


def test_minimum_distance_classifier_known() -> None:
    X = np.array([[0.0], [0.2], [9.0], [9.2]])
    y = np.array([0, 0, 1, 1])
    pred = minimum_distance_classifier(X, y, np.array([[0.1], [9.1]]))
    np.testing.assert_array_equal(pred, np.array([0, 1]))


def test_minimum_distance_classifier_validation() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="2D"):
        minimum_distance_classifier(np.array([0.0, 1.0, 2.0]), y, X)
    with pytest.raises(ValueError, match="features"):
        minimum_distance_classifier(X, y, np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="length"):
        minimum_distance_classifier(X, np.array([0, 1]), X)


# ---------------------------------------------------------------------------
# KNNClassifier
# ---------------------------------------------------------------------------


def test_knn_blobs_train_accuracy_100() -> None:
    X, y = _make_binary_blobs(n=25)
    # Đổi nhãn {-1,+1} -> {0,1} cho tiện so sánh.
    yb = (y == 1).astype(int)
    knn = KNNClassifier(k=3).fit(X, yb)
    assert accuracy_score(yb, knn.predict(X)) == 1.0


def test_knn_tie_broken_by_smaller_label() -> None:
    # Điểm test nằm chính giữa hai mẫu thuộc hai lớp khác nhau -> hòa phiếu.
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])
    knn = KNNClassifier(k=2).fit(X, y)
    assert int(knn.predict(np.array([[0.0]]))[0]) == 0


def test_knn_validation() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match=">= 1"):
        KNNClassifier(k=0)
    with pytest.raises(TypeError, match="integer"):
        KNNClassifier(k=3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="larger"):
        KNNClassifier(k=5).fit(X, y)
    knn = KNNClassifier(k=1).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        knn.predict(np.array([[0.0, 1.0]]))


# ---------------------------------------------------------------------------
# LinearSVM
# ---------------------------------------------------------------------------


def test_linear_svm_blobs_train_accuracy_100() -> None:
    X, y = _make_binary_blobs(n=25)
    svm = LinearSVM(C=1.0, lr=0.1, epochs=300, seed=0).fit(X, y)
    assert accuracy_score(y, svm.predict(X)) == 1.0


def test_linear_svm_margin_sign_convention() -> None:
    X, y = _make_binary_blobs(n=20)
    svm = LinearSVM(C=1.0, lr=0.1, epochs=300, seed=1).fit(X, y)
    # decision_function phải đúng bằng w·x - b (quy ước dấu của slide).
    np.testing.assert_allclose(svm.decision_function(X), X @ svm.w - svm.b)
    # Trên dữ liệu tách được, dấu của decision khớp nhãn và w có đúng chiều.
    assert np.all(svm.predict(X) == y)
    assert svm.w.shape == (2,)


def test_linear_svm_xor_not_solved() -> None:
    # XOR không tách được tuyến tính: một siêu phẳng chỉ phân đúng tối đa 3/4
    # điểm -> accuracy của SVM tuyến tính không thể vượt quá 0.75.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1, 1, 1, -1])
    svm = LinearSVM(C=1.0, lr=0.1, epochs=500, seed=0).fit(X, y)
    assert accuracy_score(y, svm.predict(X)) <= 0.75


def test_linear_svm_determinism() -> None:
    X, y = _make_binary_blobs(n=20)
    a = LinearSVM(C=1.0, lr=0.1, epochs=200, seed=42).fit(X, y)
    b = LinearSVM(C=1.0, lr=0.1, epochs=200, seed=42).fit(X, y)
    np.testing.assert_array_equal(a.w, b.w)
    assert a.b == b.b
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_linear_svm_loss_decreases() -> None:
    X, y = _make_binary_blobs(n=25)
    svm = LinearSVM(C=1.0, lr=0.1, epochs=300, seed=2).fit(X, y)
    assert len(svm.loss_history_) == 300
    # Với dữ liệu tách được và lr nhỏ, SGD hội tụ về cực tiểu rất nhanh rồi đứng
    # yên (plateau), nên chỉ khẳng định lỗi cuối không lớn hơn lỗi đầu. (Kiểm tra
    # "loss giảm dần" thực sự nằm ở test MLP phía dưới.)
    assert svm.loss_history_[-1] <= svm.loss_history_[0]


def test_linear_svm_validation() -> None:
    X = np.array([[-1.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        LinearSVM().fit(X, np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="positive"):
        LinearSVM(lr=0.0)
    with pytest.raises(TypeError, match="integer"):
        LinearSVM(seed=1.5)  # type: ignore[arg-type]
    svm = LinearSVM(lr=0.1, epochs=50).fit(X, np.array([-1, 1, 1]))
    with pytest.raises(ValueError, match="features"):
        svm.predict(np.array([[0.0, 1.0]]))


def test_one_vs_rest_svm_multiclass() -> None:
    X, y = _make_multiclass_blobs(n=20)
    ovr = OneVsRestSVM(C=1.0, lr=0.1, epochs=300, seed=3).fit(X, y)
    assert accuracy_score(y, ovr.predict(X)) == 1.0
    assert ovr.decision_function(X).shape == (60, 3)
    assert len(ovr.estimators_) == 3  # type: ignore[arg-type]


def test_one_vs_rest_svm_validation() -> None:
    X = np.array([[-1.0], [0.0], [1.0]])
    with pytest.raises(ValueError, match="2 classes"):
        OneVsRestSVM().fit(X, np.array([0, 0, 0]))


# ---------------------------------------------------------------------------
# MLPClassifier
# ---------------------------------------------------------------------------


def test_mlp_xor_solved() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    mlp = MLPClassifier(hidden=(4,), lr=1.0, epochs=2000, seed=7).fit(X, y)
    assert accuracy_score(y, mlp.predict(X)) == 1.0


def test_mlp_blobs_train_accuracy_100() -> None:
    X, y = _make_binary_blobs(n=20)
    yb = (y == 1).astype(int)
    mlp = MLPClassifier(hidden=(8,), lr=0.5, epochs=1000, seed=0).fit(X, yb)
    assert accuracy_score(yb, mlp.predict(X)) == 1.0


def test_mlp_softmax_rows_sum_to_one() -> None:
    X, y = _make_binary_blobs(n=10)
    yb = (y == 1).astype(int)
    mlp = MLPClassifier(hidden=(4,), lr=0.5, epochs=100, seed=0).fit(X, yb)
    proba = mlp.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    # Hàm softmax độc lập cũng phải chuẩn hóa từng hàng.
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(softmax(z).sum(axis=1), 1.0, atol=1e-12)


def test_mlp_loss_decreases() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    mlp = MLPClassifier(hidden=(4,), lr=1.0, epochs=500, seed=7).fit(X, y)
    assert mlp.loss_history_[-1] < mlp.loss_history_[0]


def test_mlp_determinism() -> None:
    X, y = _make_binary_blobs(n=10)
    yb = (y == 1).astype(int)
    a = MLPClassifier(hidden=(6,), lr=0.5, epochs=100, seed=11).fit(X, yb)
    b = MLPClassifier(hidden=(6,), lr=0.5, epochs=100, seed=11).fit(X, yb)
    for wa, wb in zip(a.W_, b.W_):  # type: ignore[arg-type]
        np.testing.assert_array_equal(wa, wb)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_mlp_tanh_hidden() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    mlp = MLPClassifier(
        hidden=(4,), lr=1.0, epochs=2000, seed=7, activation="tanh"
    ).fit(X, y)
    assert accuracy_score(y, mlp.predict(X)) == 1.0


def test_mlp_validation() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="activation"):
        MLPClassifier(activation="relu")
    with pytest.raises(ValueError, match=">= 1"):
        MLPClassifier(hidden=(0,))
    with pytest.raises(ValueError, match="2 classes"):
        MLPClassifier().fit(X, np.array([0, 0]))
    mlp = MLPClassifier(hidden=(4,), lr=0.5, epochs=10).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        mlp.predict(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# Các khối CNN (conv2d_valid, relu, max_pool2d, flatten)
# ---------------------------------------------------------------------------


def test_conv2d_valid_hand_computed_4x4() -> None:
    # Ví dụ trong slide §7.4: filter 1 (3x3), ảnh 4x4, stride 1 -> 2x2.
    x = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]], dtype=float)
    k = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    expected = np.array([[3.0, -3.0], [-3.0, 3.0]])
    np.testing.assert_array_equal(conv2d_valid(x, k), expected)


def test_conv2d_valid_matches_numpy_reference() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 6))
    k = rng.normal(size=(3, 3))
    for stride in (1, 2):
        ours = conv2d_valid(x, k, stride)
        ref = _ref_corr2d(x, k, stride)
        np.testing.assert_allclose(ours, ref, atol=1e-12)
        assert ours.shape == ((6 - 3) // stride + 1, (6 - 3) // stride + 1)


def test_conv2d_valid_1x1_kernel() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(conv2d_valid(x, np.array([[2.0]])), 2.0 * x)


def test_conv2d_valid_validation() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = np.array([[1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="larger"):
        conv2d_valid(x, np.ones((3, 3)))
    with pytest.raises(ValueError, match=">= 1"):
        conv2d_valid(x, k, stride=0)
    with pytest.raises(ValueError, match="2D"):
        conv2d_valid(np.array([1.0, 2.0]), k)


def test_relu() -> None:
    np.testing.assert_array_equal(
        relu(np.array([-1.0, 0.0, 2.0])), np.array([0.0, 0.0, 2.0])
    )


def test_max_pool2d_shape_and_values() -> None:
    x = np.arange(1, 17, dtype=float).reshape(4, 4)
    out = max_pool2d(x, 2, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == 6.0  # max của [[1,2],[5,6]]
    assert out[1, 1] == 16.0  # max của [[11,12],[15,16]]
    # stride=1 -> (4-2)//1 + 1 = 3
    assert max_pool2d(x, 2, 1).shape == (3, 3)


def test_max_pool2d_validation() -> None:
    with pytest.raises(ValueError, match="larger"):
        max_pool2d(np.array([[1.0, 2.0]]), 3, 1)
    with pytest.raises(ValueError, match=">= 1"):
        max_pool2d(np.ones((4, 4)), 2, 0)


def test_flatten() -> None:
    np.testing.assert_array_equal(
        flatten(np.array([[1, 2], [3, 4]])), np.array([1, 2, 3, 4])
    )
    x3 = np.arange(8).reshape(2, 2, 2)
    assert flatten(x3).shape == (8,)


# ---------------------------------------------------------------------------
# accuracy_score và confusion_matrix
# ---------------------------------------------------------------------------


def test_accuracy_score() -> None:
    assert accuracy_score(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) == 0.75


def test_confusion_matrix() -> None:
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]))
    np.testing.assert_array_equal(cm, np.array([[1, 1], [0, 1]]))
    # Với labels tường minh, thứ tự hàng/cột theo labels đã cho.
    cm2 = confusion_matrix(np.array([2, 0]), np.array([2, 0]), labels=np.array([0, 2]))
    np.testing.assert_array_equal(cm2, np.array([[1, 0], [0, 1]]))


def test_metrics_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        accuracy_score(np.array([0, 1]), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="shape"):
        confusion_matrix(np.array([0, 1]), np.array([0, 1, 1]))
