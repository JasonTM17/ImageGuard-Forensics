"""Các bộ phân lớp cho học máy cơ bản (NumPy thuần túy, không phụ thuộc sklearn).

Mô-đun này hiện thực Chương 7 "Pattern Recognition" (slide Xử lý ảnh số,
giảng viên dunghv, HCMUTE):

- §7.1  Giới thiệu — patterns và pattern classes, biểu diễn đặc trưng bằng
        vector: ``minimum_distance_classifier`` (bộ phân lớp trung bình gần nhất).
- §7.2  Support Vector Machine — siêu phẳng biên cực đại (maximum margin),
        biên mềm (soft margin) với tham số C: ``LinearSVM`` và ``OneVsRestSVM``
        cho bài toán đa lớp.
- §7.3  Neural Network — ước lượng lỗi (RMSE), điều chỉnh trọng số bằng lan
        truyền ngược (back-propagation): ``MLPClassifier``.
- §7.4  Deep CNN — convolution, stride, max pooling, flatten: các khối
        ``conv2d_valid``, ``relu``, ``max_pool2d``, ``flatten`` (chỉ minh họa,
        không huấn luyện).

Đây là bộ công cụ DEMO/HỌC TẬP: chỉ dùng để minh họa thuật toán. KHÔNG import
vào detector giám định thật và KHÔNG đưa ra bất kỳ tuyên bố nào về độ chính
xác trên dữ liệu giám định ảnh.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KNNClassifier",
    "LinearSVM",
    "MLPClassifier",
    "OneVsRestSVM",
    "accuracy_score",
    "confusion_matrix",
    "conv2d_valid",
    "flatten",
    "max_pool2d",
    "minimum_distance_classifier",
    "relu",
    "softmax",
]


# ---------------------------------------------------------------------------
# Các hàm kiểm tra dữ liệu đầu vào (giữ phong cách với threshold.py/edges.py)
# ---------------------------------------------------------------------------


def _check_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Kiểm tra tham số nguyên: từ chối bool và kiểu không nguyên (TypeError)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    v = int(value)
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return v


def _check_positive_float(value: object, name: str) -> float:
    """Kiểm tra tham số số thực dương và hữu hạn."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return v


def _check_seed(seed: object) -> int:
    """Kiểm tra seed: chỉ nhận số nguyên (dùng cho numpy.random.default_rng)."""
    return _check_int(seed, "seed")


def _as_float_matrix(X: object, name: str = "X") -> np.ndarray:
    """Chuyển đầu vào thành ma trận float64 (n_samples, n_features)."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (n_samples, n_features)")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric")
    return arr.astype(np.float64, copy=False)


def _as_label_vector(y: object, n: int, name: str = "y") -> np.ndarray:
    """Kiểm tra vector nhãn 1D có độ dài khớp với số mẫu."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n} (got {arr.shape[0]})")
    return arr


def _majority_vote(votes: np.ndarray) -> object:
    """Bỏ phiếu đa số; khi hòa thì chọn nhãn nhỏ nhất (theo thứ tự sắp xếp)."""
    labels, counts = np.unique(votes, return_counts=True)
    top = labels[counts == counts.max()]
    # np.unique trả về đã sắp xếp tăng dần => top[0] là nhãn nhỏ nhất khi hòa.
    return top[0]


# ---------------------------------------------------------------------------
# §7.1 — Bộ phân lớp trung bình gần nhất (minimum distance classifier)
# ---------------------------------------------------------------------------


def minimum_distance_classifier(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """Phân lớp theo khoảng cách đến trung bình từng lớp (nearest class mean).

    Đây là dạng ra quyết định pattern-class đơn giản nhất trong slide: với mỗi
    lớp, ta tính vector trung bình (prototype); một mẫu mới được gán vào lớp có
    trung bình gần nó nhất theo khoảng cách Euclid.

    Công thức (bằng lời): ``argmin_k || x - mean(X[y == k]) ||``.

    Tham số:
        X: ma trận huấn luyện (n_samples, n_features).
        y: nhãn huấn luyện, 1D độ dài n_samples.
        X_test: ma trận cần dự đoán (n_test, n_features).

    Trả về:
        np.ndarray nhãn dự đoán (n_test,) có cùng kiểu với ``y``.

    Ví dụ:
        >>> import numpy as np
        >>> X = np.array([[0.0], [0.2], [9.0], [9.2]])
        >>> y = np.array([0, 0, 1, 1])
        >>> minimum_distance_classifier(X, y, np.array([[0.1], [9.1]]))
        array([0, 1])
    """
    Xtr = _as_float_matrix(X, "X")
    ytr = _as_label_vector(y, Xtr.shape[0], "y")
    Xte = _as_float_matrix(X_test, "X_test")
    if Xte.shape[1] != Xtr.shape[1]:
        raise ValueError("X_test must have the same number of features as X")

    classes = np.unique(ytr)
    # Trung bình từng lớp: prototype đại diện cho mỗi pattern class.
    means = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes])
    # Khoảng cách Euclid bình phương từ mỗi mẫu test đến từng prototype.
    dist = ((Xte[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    return classes[np.argmin(dist, axis=1)]


# ---------------------------------------------------------------------------
# KNN — k láng giềng gần nhất
# ---------------------------------------------------------------------------


class KNNClassifier:
    """Bộ phân lớp k láng giềng gần nhất (k-Nearest Neighbors).

    Quyết định nhãn của một mẫu bằng cách bỏ phiếu đa số trong ``k`` mẫu huấn
    luyện gần nó nhất theo khoảng cách Euclid. Khi hòa phiếu, nhãn nhỏ hơn
    thắng (theo thứ tự sắp xếp).

    Tham số:
        k: số láng giềng xét đến (nguyên dương).

    Ví dụ:
        >>> import numpy as np
        >>> X = np.array([[0.0], [0.1], [10.0], [10.1]])
        >>> y = np.array([0, 0, 1, 1])
        >>> knn = KNNClassifier(k=3).fit(X, y)
        >>> knn.predict(np.array([[0.05], [9.9]]))
        array([0, 1])
    """

    def __init__(self, k: int = 3) -> None:
        self.k = _check_int(k, "k", minimum=1)
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        """Lưu tập huấn luyện (kNN không học tham số, chỉ ghi nhớ dữ liệu)."""
        Xa = _as_float_matrix(X, "X")
        ya = _as_label_vector(y, Xa.shape[0], "y")
        if self.k > Xa.shape[0]:
            raise ValueError("k cannot be larger than the number of samples")
        self.X_ = Xa.copy()
        self.y_ = ya.copy()
        self._n_features = Xa.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Trả nhãn dự đoán cho từng mẫu trong ``X`` (bỏ phiếu k láng giềng)."""
        Xa = _as_float_matrix(X, "X")
        if self.X_ is None or self.y_ is None:
            raise ValueError("model must be fitted before predict")
        if Xa.shape[1] != self._n_features:
            raise ValueError("X must have the same number of features as fit")

        # Khoảng cách Euclid bình phương giữa mọi cặp (test, train).
        diff = Xa[:, None, :] - self.X_[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        # Lấy chỉ số của k mẫu gần nhất theo thứ tự khoảng cách tăng dần.
        nn = np.argsort(dist, axis=1)[:, : self.k]

        out = np.empty(Xa.shape[0], dtype=self.y_.dtype)
        for i in range(Xa.shape[0]):
            out[i] = _majority_vote(self.y_[nn[i]])
        return out


# ---------------------------------------------------------------------------
# §7.2 — Support Vector Machine tuyến tính (biên cực đại / biên mềm)
# ---------------------------------------------------------------------------


class LinearSVM:
    """SVM tuyến tính nhị phân huấn luyện bằng dưới-đạo hàm (sub-gradient).

    Ý tưởng biên (margin) theo slide §7.2: tìm siêu phẳng ``w·x - b = 0`` tách
    hai lớp ``{-1, +1}`` sao cho khoảng cách (biên) từ siêu phẳng đến các điểm
    gần nhất là lớn nhất. Với biên mềm (soft margin), ta cho phép một số điểm
    vi phạm biên và thêm tham số C đánh đổi giữa biên rộng và số lỗi.

    Hàm mục tiêu (cực tiểu hóa):
        ``J(w, b) = (1/2)||w||^2 + C * mean(max(0, 1 - y_i(w·x_i - b)))``

    Quy ước dấu biên: ``decision_function(x) = w·x - b``; dự đoán
    ``+1`` nếu giá trị này ``>= 0``, ngược lại ``-1``. Tham số chuẩn hóa
    ``lambda`` trong một số tài liệu chính là ``1 / C``.

    Tham số:
        C: hệ số đánh đổi biên mềm (C càng lớn → ít chuẩn hóa, biên cứng hơn).
        lr: tốc độ học (learning rate) cho dưới-đạo hàm.
        epochs: số vòng lặp duyệt toàn bộ dữ liệu.
        seed: hạt ngẫu nhiên để khởi tạo trọng số và xáo trộn (tái lập được).

    Thuộc tính sau khi fit:
        w: vector pháp tuyến siêu phẳng (n_features,).
        b: offset theo quy ước ``w·x - b = 0``.
        loss_history_: danh sách giá trị hàm mất mát sau mỗi epoch.

    Ví dụ:
        >>> import numpy as np
        >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        >>> y = np.array([-1, -1, 1, 1])
        >>> svm = LinearSVM(C=1.0, lr=0.1, epochs=200, seed=0).fit(X, y)
        >>> svm.predict(np.array([[-3.0], [3.0]]))
        array([-1,  1])
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        epochs: int = 500,
        seed: int = 0,
    ) -> None:
        self.C = _check_positive_float(C, "C")
        self.lr = _check_positive_float(lr, "lr")
        self.epochs = _check_int(epochs, "epochs", minimum=1)
        self.seed = _check_seed(seed)
        self.w: np.ndarray | None = None
        self.b: float | None = None
        self.loss_history_: list[float] = []

    def _check_input(self, X: np.ndarray) -> np.ndarray:
        """Kiểm tra ma trận đầu vào khớp số đặc trưng của mô hình đã fit."""
        if self.w is None:
            raise ValueError("model must be fitted before predict")
        Xa = _as_float_matrix(X, "X")
        if Xa.shape[1] != self.w.shape[0]:
            raise ValueError("X must have the same number of features as fit")
        return Xa

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSVM:
        """Huấn luyện bằng SGD trên dưới-đạo hàm của hinge loss + L2."""
        Xa = _as_float_matrix(X, "X")
        ya = _as_label_vector(y, Xa.shape[0], "y")
        uniq = np.unique(ya)
        # Yêu cầu nhãn nhị phân đúng hai lớp trong {-1, +1} (quy ước của slide).
        if uniq.size != 2 or not set(uniq.tolist()) <= {-1.0, 1.0}:
            raise ValueError("y must contain exactly the two labels {-1, +1}")

        rng = np.random.default_rng(self.seed)
        n_samples, n_features = Xa.shape
        # Khởi tạo w nhỏ gần 0; b bắt đầu bằng 0.
        w = rng.standard_normal(n_features) * 0.01
        b = 0.0

        self.loss_history_ = []
        for _ in range(self.epochs):
            # Xáo trộn thứ tự để SGD không bị lệch theo thứ tự dữ liệu cố định.
            order = rng.permutation(n_samples)
            for i in order:
                xi = Xa[i]
                yi = ya[i]
                # Biên hình học của mẫu i: y_i * (w·x_i - b).
                margin = yi * (float(xi @ w) - b)
                if margin < 1.0:
                    # Vi phạm biên → cập nhật theo dưới-đạo hàm của hinge loss:
                    #   dJ/dw = w - C*y_i*x_i  và  dJ/db = C*y_i.
                    w = (1.0 - self.lr) * w + self.lr * self.C * yi * xi
                    b = b - self.lr * self.C * yi
            # Hàm mất mát toàn phần: chuẩn hóa L2 + trung bình hinge loss.
            hinge = np.maximum(0.0, 1.0 - ya * (Xa @ w - b))
            loss = 0.5 * float(w @ w) + self.C * float(np.mean(hinge))
            self.loss_history_.append(loss)

        self.w = w
        self.b = float(b)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Giá trị quyết định ``w·x - b`` cho từng mẫu (dấu quyết định lớp)."""
        Xa = self._check_input(X)
        return Xa @ self.w - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán: ``+1`` nếu ``decision_function >= 0``, ngược lại ``-1``."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1)


class OneVsRestSVM:
    """SVM đa lớp theo chiến lược one-vs-rest (một lớp chống lại phần còn lại).

    Với mỗi lớp ``c``, ta huấn luyện một ``LinearSVM`` nhị phân: nhãn ``+1`` cho
    các mẫu thuộc ``c`` và ``-1`` cho phần còn lại. Mẫu mới được gán vào lớp có
    ``decision_function`` lớn nhất.

    Tham số:
        C, lr, epochs, seed: truyền nguyên vẹn cho từng ``LinearSVM`` con.

    Thuộc tính sau khi fit:
        classes_: các nhãn lớp (đã sắp xếp).
        estimators_: danh sách các ``LinearSVM`` nhị phân.

    Ví dụ:
        >>> import numpy as np
        >>> X = np.array([[-2.0], [-1.0], [0.5], [2.0]])
        >>> y = np.array([0, 0, 1, 2])
        >>> ovr = OneVsRestSVM(C=1.0, lr=0.1, epochs=300, seed=0).fit(X, y)
        >>> ovr.predict(np.array([[-3.0], [0.5], [3.0]]))
        array([0, 1, 2])
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        epochs: int = 500,
        seed: int = 0,
    ) -> None:
        self.C = _check_positive_float(C, "C")
        self.lr = _check_positive_float(lr, "lr")
        self.epochs = _check_int(epochs, "epochs", minimum=1)
        self.seed = _check_seed(seed)
        self.classes_: np.ndarray | None = None
        self.estimators_: list[LinearSVM] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> OneVsRestSVM:
        """Huấn luyện một SVM nhị phân cho từng lớp."""
        Xa = _as_float_matrix(X, "X")
        ya = _as_label_vector(y, Xa.shape[0], "y")
        self.classes_ = np.unique(ya)
        if self.classes_.size < 2:
            raise ValueError("OneVsRestSVM needs at least 2 classes")

        self.estimators_ = []
        for c in self.classes_:
            # Nhãn nhị phân: +1 cho lớp hiện tại, -1 cho phần còn lại.
            binary = np.where(ya == c, 1.0, -1.0)
            svm = LinearSVM(C=self.C, lr=self.lr, epochs=self.epochs, seed=self.seed)
            svm.fit(Xa, binary)
            self.estimators_.append(svm)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Ma trận điểm (n_samples, n_classes): cột ứng với từng SVM con."""
        if self.estimators_ is None:
            raise ValueError("model must be fitted before decision_function")
        return np.column_stack([svm.decision_function(X) for svm in self.estimators_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán: lớp có điểm quyết định lớn nhất."""
        if self.classes_ is None:
            raise ValueError("model must be fitted before predict")
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


# ---------------------------------------------------------------------------
# Các hàm kích hoạt và softmax (dùng chung cho MLP)
# ---------------------------------------------------------------------------


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax ổn định số học: mỗi hàng/trục trở thành phân phối xác suất.

    Công thức: ``exp(z - max(z)) / sum(exp(z - max(z)))`` trên ``axis``. Việc
    trừ ``max`` giúp tránh tràn số khi ``z`` chứa giá trị lớn.

    Ví dụ:
        >>> softmax(np.array([[2.0, 1.0]]))
        array([[0.731..., 0.268...]])
    """
    arr = np.asarray(z, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("z must be non-empty")
    zmax = np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr - zmax)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Hàm sigmoid (unipolar continuous theo slide §7.3): 1/(1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """Đạo hàm sigmoid tính theo giá trị đã kích hoạt: a * (1 - a)."""
    return a * (1.0 - a)


def _tanh(z: np.ndarray) -> np.ndarray:
    """Hàm tanh (bipolar continuous theo slide): tanh(z)."""
    return np.tanh(z)


def _tanh_deriv(a: np.ndarray) -> np.ndarray:
    """Đạo hàm tanh tính theo giá trị đã kích hoạt: 1 - a^2."""
    return 1.0 - a * a


# ---------------------------------------------------------------------------
# §7.3 — Mạng nơ-ron nhiều lớp (MLP) + lan truyền ngược
# ---------------------------------------------------------------------------


class MLPClassifier:
    """Mạng nơ-ron truyền thẳng nhỏ, huấn luyện bằng lan truyền ngược.

    Kiến trúc: ``n_features -> hidden... -> n_classes``. Các lớp ẩn dùng hàm
    kích hoạt sigmoid hoặc tanh; lớp ra dùng softmax. Huấn luyện theo đúng các
    bước của slide §7.3:

    1. Lan truyền thuận (forward pass) — tính tổ hợp tuyến tính có trọng số rồi
       đưa qua hàm kích hoạt.
    2. Ước lượng lỗi (error estimation) — cross-entropy giữa đầu ra và nhãn;
       với softmax, lỗi lớp ra rút gọn thành ``(out - y)`` (chính là hiệu "thực
       tế trừ kỳ vọng", ngược dấu với ``NErrorFactor`` của slide).
    3. Lan truyền ngược (back-propagation) — truyền lỗi về lớp ẩn nhân với đạo
       hàm kích hoạt ``a*(1-a)`` (đúng luật delta ``Nout*(1-Nout)`` của slide).
    4. Điều chỉnh trọng số (weights adjusting) — ``W -= lr * dW`` theo gradient
       trung bình trên toàn batch.

    Tham số:
        hidden: kích thước các lớp ẩn (int hoặc tuple các int dương).
        lr: tốc độ học.
        epochs: số vòng lặp (mỗi vòng duyệt toàn bộ dữ liệu — batch GD).
        seed: hạt ngẫu nhiên cho khởi tạo trọng số (tái lập được).
        activation: "sigmoid" (mặc định) hoặc "tanh" cho lớp ẩn.

    Thuộc tính sau khi fit:
        classes_: các nhãn lớp (đã sắp xếp).
        W_, b_: danh sách ma trận trọng số và vector bias từng lớp.
        loss_history_: giá trị cross-entropy sau mỗi epoch.

    Ví dụ (hàm XOR — bài toán phi tuyến mà MLP giải được):
        >>> import numpy as np
        >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        >>> y = np.array([0, 1, 1, 0])
        >>> mlp = MLPClassifier(hidden=(4,), lr=1.0, epochs=1500, seed=7).fit(X, y)
        >>> mlp.predict(X)
        array([0, 1, 1, 0])
    """

    def __init__(
        self,
        hidden: int | tuple[int, ...] = (16,),
        lr: float = 0.5,
        epochs: int = 1000,
        seed: int = 0,
        activation: str = "sigmoid",
    ) -> None:
        if isinstance(hidden, (int, np.integer)) and not isinstance(hidden, bool):
            hidden = (int(hidden),)
        self.hidden: tuple[int, ...] = tuple(
            _check_int(h, "hidden", minimum=1) for h in hidden
        )
        self.lr = _check_positive_float(lr, "lr")
        self.epochs = _check_int(epochs, "epochs", minimum=1)
        self.seed = _check_seed(seed)
        if activation not in ("sigmoid", "tanh"):
            raise ValueError("activation must be 'sigmoid' or 'tanh'")
        self.activation = activation
        if activation == "sigmoid":
            self._act = _sigmoid
            self._act_deriv = _sigmoid_deriv
        else:
            self._act = _tanh
            self._act_deriv = _tanh_deriv

        self.classes_: np.ndarray | None = None
        self.W_: list[np.ndarray] | None = None
        self.b_: list[np.ndarray] | None = None
        self.loss_history_: list[float] = []
        self._n_features: int | None = None

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Lan truyền thuận: trả xác suất softmax (n_samples, n_classes)."""
        if self.W_ is None or self.b_ is None:
            raise ValueError("model must be fitted before predict")
        a = X
        n_layers = len(self.W_)
        for i in range(n_layers):
            z = a @ self.W_[i] + self.b_[i]
            # Lớp cuối dùng softmax, lớp ẩn dùng hàm kích hoạt đã chọn.
            a = softmax(z) if i == n_layers - 1 else self._act(z)
        return a

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        """Huấn luyện mạng bằng gradient descent toàn batch + back-propagation."""
        Xa = _as_float_matrix(X, "X")
        n_samples, n_features = Xa.shape
        ya = _as_label_vector(y, n_samples, "y")
        self.classes_ = np.unique(ya)
        n_classes = self.classes_.size
        if n_classes < 2:
            raise ValueError("MLPClassifier needs at least 2 classes")
        self._n_features = n_features

        # Mã hóa nhãn thành one-hot: mục tiêu "kỳ vọng" cho lớp ra.
        y_idx = np.searchsorted(self.classes_, ya)
        Y = np.eye(n_classes, dtype=np.float64)[y_idx]

        # Khởi tạo trọng số nhỏ ngẫu nhiên trong [-1, 1] (theo slide "Initial
        # weights"), bias khởi tạo bằng 0.
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden, n_classes]
        W: list[np.ndarray] = []
        b: list[np.ndarray] = []
        for layer in range(len(sizes) - 1):
            n_in, n_out = sizes[layer], sizes[layer + 1]
            W.append(rng.uniform(-1.0, 1.0, size=(n_in, n_out)))
            b.append(np.zeros(n_out, dtype=np.float64))

        n_layers = len(W)
        self.loss_history_ = []
        for _ in range(self.epochs):
            # ---------- 1) LAN TRUYỀN THUẬN ----------
            a = Xa
            acts = [a]  # acts[i] = đầu ra của lớp i-1 (acts[0] là input)
            zs: list[np.ndarray] = []
            for i in range(n_layers):
                z = a @ W[i] + b[i]  # tổ hợp tuyến tính có trọng số
                zs.append(z)
                a = softmax(z) if i == n_layers - 1 else self._act(z)
                acts.append(a)
            out = acts[-1]

            # ---------- 2) ƯỚC LƯỢNG LỖI ----------
            # Cross-entropy; cộng epsilon để tránh log(0).
            loss = -np.mean(np.sum(Y * np.log(out + 1e-12), axis=1))
            self.loss_history_.append(float(loss))

            # ---------- 3) LAN TRUYỀN NGƯỢC ----------
            # Lỗi lớp ra: với softmax + cross-entropy, delta = (out - Y).
            delta = out - Y
            gradW: list[np.ndarray] = [np.empty_like(w) for w in W]
            gradb: list[np.ndarray] = [np.empty_like(bb) for bb in b]
            for i in reversed(range(n_layers)):
                # Gradient trung bình theo trọng số và bias của lớp i.
                gradW[i] = acts[i].T @ delta / n_samples
                gradb[i] = np.sum(delta, axis=0) / n_samples
                if i > 0:
                    # Truyền lỗi ngược qua lớp ẩn: delta * W^T * đạo hàm kích
                    # hoạt (đúng luật delta Nout*(1-Nout) của slide).
                    delta = (delta @ W[i].T) * self._act_deriv(acts[i])

            # ---------- 4) ĐIỀU CHỈNH TRỌNG SỐ ----------
            for i in range(n_layers):
                W[i] = W[i] - self.lr * gradW[i]
                b[i] = b[i] - self.lr * gradb[i]

        self.W_ = W
        self.b_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất softmax cho từng lớp (n_samples, n_classes)."""
        Xa = _as_float_matrix(X, "X")
        if self._n_features is None or Xa.shape[1] != self._n_features:
            raise ValueError("X must have the same number of features as fit")
        return self._forward(Xa)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán: lớp có xác suất softmax lớn nhất."""
        if self.classes_ is None:
            raise ValueError("model must be fitted before predict")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# §7.4 — Các khối CNN nhỏ (chỉ minh họa, không huấn luyện)
# ---------------------------------------------------------------------------


def conv2d_valid(x: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """Tích chập "valid" (không đệm, không lật kernel) giữa ảnh và bộ lọc.

    Thực hiện đúng phép "dot product" từng vùng như slide §7.4: trượt kernel
    trên ảnh với bước ``stride`` và chỉ giữ các vị trí mà kernel nằm trọn trong
    ảnh. Chú ý: giống các framework CNN, ta dùng tương quan chéo (không lật
    kernel), nên kết quả là ``sum(patch * kernel)``.

    Công thức kích thước ra: ``(H - kh)//stride + 1`` và ``(W - kw)//stride + 1``.

    Tham số:
        x: ảnh 2D (H, W).
        kernel: bộ lọc 2D (kh, kw).
        stride: bước trượt (nguyên dương).

    Trả về:
        np.ndarray (oh, ow) float64.

    Ví dụ (bộ lọc 1 của slide, kernel 3x3, ảnh 4x4, stride 1 → 2x2):
        >>> import numpy as np
        >>> x = np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[1,0,0,1]], float)
        >>> k = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
        >>> conv2d_valid(x, k)
        array([[ 3., -3.],
               [-3.,  3.]])
    """
    xr = _as_float_matrix(x, "x")
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.size == 0:
        raise ValueError("kernel must be a non-empty 2D array")
    s = _check_int(stride, "stride", minimum=1)

    H, W = xr.shape
    kh, kw = k.shape
    if kh > H or kw > W:
        raise ValueError("kernel must not be larger than x")
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1

    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            # Trượt kernel theo stride và cộng tích từng phần tử (dot product).
            patch = xr[i * s : i * s + kh, j * s : j * s + kw]
            out[i, j] = np.sum(patch * k)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    """Hàm kích hoạt ReLU theo từng phần tử: ``max(0, x)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("x must be non-empty")
    return np.maximum(arr, 0.0)


def max_pool2d(x: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Max pooling 2D: lấy giá trị lớn nhất trong từng cửa sổ ``size x size``.

    Giảm kích thước ảnh đặc trưng bằng cách giữ giá trị lớn nhất mỗi vùng, giúp
    giảm số tham số nhưng vẫn giữ đặc trưng nổi bật (theo slide §7.4).

    Công thức kích thước ra: ``(H - size)//stride + 1`` và ``(W - size)//stride + 1``.

    Tham số:
        x: ảnh 2D (H, W).
        size: kích thước cửa sổ vuông (nguyên dương).
        stride: bước trượt (nguyên dương).

    Trả về:
        np.ndarray (oh, ow) float64.

    Ví dụ:
        >>> import numpy as np
        >>> max_pool2d(np.array([[1.,2.],[3.,4.]]), 2, 2)
        array([[4.]])
    """
    xr = _as_float_matrix(x, "x")
    sz = _check_int(size, "size", minimum=1)
    s = _check_int(stride, "stride", minimum=1)

    H, W = xr.shape
    if sz > H or sz > W:
        raise ValueError("size must not be larger than x")
    oh = (H - sz) // s + 1
    ow = (W - sz) // s + 1

    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            # Giữ giá trị lớn nhất trong cửa sổ hiện tại.
            out[i, j] = np.max(xr[i * s : i * s + sz, j * s : j * s + sz])
    return out


def flatten(x: np.ndarray) -> np.ndarray:
    """Trải phẳng mảng nhiều chiều thành vector 1D (chuẩn bị cho lớp FC).

    Theo slide §7.4, các feature map sau pooling được "flattened" rồi đưa vào
    mạng truyền thẳng. Hàm chỉ đổi hình dạng, giữ nguyên thứ tự phần tử (C-order).

    Ví dụ:
        >>> flatten(np.array([[1, 2], [3, 4]]))
        array([1, 2, 3, 4])
    """
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError("x must be non-empty")
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Các hàm đánh giá (metrics)
# ---------------------------------------------------------------------------


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tỷ lệ dự đoán đúng: ``mean(y_true == y_pred)`` trong [0, 1]."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1D arrays")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> np.ndarray:
    """Ma trận nhầm lẫn: hàng là nhãn thật, cột là nhãn dự đoán.

    Tham số:
        y_true, y_pred: hai vector nhãn cùng độ dài.
        labels: danh sách nhãn dùng làm thứ tự hàng/cột (mặc định: hợp đã sắp xếp).

    Trả về:
        np.ndarray (K, K) kiểu int64; ``cm[i, j]`` là số mẫu nhãn thật ``labels[i]``
        bị dự đoán thành ``labels[j]``.

    Ví dụ:
        >>> confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]))
        array([[1, 1],
               [0, 1]])
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1D arrays")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    else:
        labels = np.asarray(labels)
    n = labels.size
    cm = np.zeros((n, n), dtype=np.int64)
    # Đếm trực tiếp từng cặp (nhãn thật, nhãn dự đoán).
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            cm[i, j] = int(np.sum((yt == true_lbl) & (yp == pred_lbl)))
    return cm
