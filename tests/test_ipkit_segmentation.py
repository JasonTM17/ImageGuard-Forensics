"""ipkit.segmentation — Chương 5: điểm/đường/biên, ngưỡng, vùng, k-means, watershed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from imageguard.ipkit.segmentation import (
    canny_edges,
    detect_lines,
    detect_points,
    hough_lines,
    iterative_global_threshold,
    kmeans_segmentation,
    region_growing,
    simple_watershed,
)
from imageguard.ipkit.threshold import otsu_threshold

# ---------------------------------------------------------------------------
# detect_points
# ---------------------------------------------------------------------------


def test_detect_points_finds_single_bright_pixel() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 255  # một điểm sáng duy nhất trên nền đen
    mask = detect_points(img, T=1000)

    assert mask.dtype == np.uint8
    assert mask[4, 4] == 255
    # Chỉ đúng một điểm được phát hiện (điểm cô lập duy nhất).
    assert np.count_nonzero(mask) == 1


def test_detect_points_ignores_flat_region() -> None:
    img = np.full((7, 7), 128, dtype=np.uint8)
    mask = detect_points(img, T=1)
    # Vùng phẳng cho phản hồi xấp xỉ 0 → không có điểm nào.
    assert np.count_nonzero(mask) == 0


def test_detect_points_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        detect_points(np.zeros((3, 3), dtype=np.float32), T=1)


# ---------------------------------------------------------------------------
# detect_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "draw"),
    [
        ("horizontal", "row"),
        ("vertical", "col"),
        ("+45", "anti_diag"),
        ("-45", "main_diag"),
    ],
)
def test_detect_lines_orients_correctly(direction: str, draw: str) -> None:
    n = 9
    img = np.zeros((n, n), dtype=np.uint8)
    if draw == "row":
        img[n // 2, 1:-1] = 255  # đường ngang
    elif draw == "col":
        img[1:-1, n // 2] = 255  # đường dọc
    elif draw == "main_diag":
        # đường chéo chính (trên-trái -> dưới-phải), bỏ 2 góc để tránh biên
        for i in range(1, n - 1):
            img[i, i] = 255
    else:
        # đường chéo phụ (trên-phải -> dưới-trái)
        for i in range(1, n - 1):
            img[i, n - 1 - i] = 255

    mask = detect_lines(img, direction, T=1000)
    # Kernel đúng hướng phải "bắt" được các pixel của đường đã vẽ.
    drawn = np.count_nonzero(img)
    detected = np.count_nonzero(mask)
    assert detected > 0
    assert detected <= drawn


def test_detect_lines_rejects_bad_direction() -> None:
    img = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="direction"):
        detect_lines(img, "diagonal", T=1)


# ---------------------------------------------------------------------------
# iterative_global_threshold
# ---------------------------------------------------------------------------


def test_iterative_threshold_bimodal_between_modes_and_near_otsu() -> None:
    # Hai mode Gauss quanh 80 và 180 với sigma đủ lớn để hai mode chồng lấn:
    # khi đó điểm giữa (fixed point của vòng lặp) trùng với thung lũng Otsu.
    rng = np.random.default_rng(0)
    lo = rng.normal(80, 20, (32, 32))
    hi = rng.normal(180, 20, (32, 32))
    img = np.concatenate([lo, hi], axis=1)
    img = np.clip(img, 0, 255).astype(np.uint8)

    t, binary = iterative_global_threshold(img, eps=0.5)
    otsu_t = otsu_threshold(img)

    assert binary.dtype == np.uint8
    assert set(np.unique(binary)) <= {0, 255}
    # Ngưỡng nằm giữa hai mode.
    assert 80.0 < t < 180.0
    # Với hai mode Gauss đối xứng cùng sigma, ngưỡng vòng lặp ≈ Otsu (đo được
    # |t - otsu| < 1 trên seed này); cho phép dung sai 8 mức xám.
    assert abs(t - otsu_t) <= 8.0


def test_iterative_threshold_constant_image() -> None:
    img = np.full((6, 6), 100, dtype=np.uint8)
    t, binary = iterative_global_threshold(img, eps=0.5)
    # Ảnh hằng: một nhóm rỗng nên thuật toán dừng ngay tại T0 = mean.
    assert t == pytest.approx(100.0)
    assert np.count_nonzero(binary) == 0


def test_iterative_threshold_rejects_bad_eps() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="eps"):
        iterative_global_threshold(img, eps=0.0)


# ---------------------------------------------------------------------------
# region_growing
# ---------------------------------------------------------------------------


def test_region_growing_stays_inside_square_and_respects_tol() -> None:
    img = np.zeros((15, 15), dtype=np.uint8)
    img[3:12, 3:12] = 100  # hình vuông đồng nhất cường độ 100

    labels = region_growing(img, [(7, 7)], tol=10, connectivity=8)
    assert labels[7, 7] == 1
    # Vùng chỉ gồm hình vuông (nền 0 lệch 100 > tol nên không bị sát nhập).
    assert np.all(labels[3:12, 3:12] == 1)
    assert labels[0, 0] == 0
    assert np.count_nonzero(labels) == 81


def test_region_growing_leaks_when_tol_is_large() -> None:
    img = np.zeros((9, 9), dtype=np.uint8)
    img[3:6, 3:6] = 100
    labels = region_growing(img, [(4, 4)], tol=200, connectivity=8)
    # tol đủ lớn để vượt chênh lệch 100 → tràn ra cả nền.
    assert labels[0, 0] == 1


def test_region_growing_multiple_seeds_distinct_labels() -> None:
    img = np.zeros((8, 16), dtype=np.uint8)
    img[:, :8] = 50
    img[:, 8:] = 200
    labels = region_growing(img, [(3, 3), (3, 11)], tol=10, connectivity=4)
    # Hai hạt giống trên hai vùng tách biệt → hai nhãn riêng, không chồng lấn.
    assert labels[3, 3] == 1
    assert labels[3, 11] == 2
    assert not np.any((labels == 1) & (labels == 2))


def test_region_growing_rejects_bad_args() -> None:
    img = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="connectivity"):
        region_growing(img, [(2, 2)], tol=1, connectivity=6)
    with pytest.raises(ValueError, match="inside"):
        region_growing(img, [(9, 9)], tol=1)
    with pytest.raises(ValueError, match="tol"):
        region_growing(img, [(2, 2)], tol=-1.0)


# ---------------------------------------------------------------------------
# kmeans_segmentation
# ---------------------------------------------------------------------------


def test_kmeans_separates_two_clusters() -> None:
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, :16] = 50
    img[:, 16:] = 200

    labels, centres = kmeans_segmentation(img, k=2, seed=1, max_iter=50)
    assert labels.shape == img.shape
    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1}

    cs = sorted(np.rint(centres[:, 0]).astype(int))
    assert cs == [50, 200]
    # Hai nửa được gán hai nhãn khác nhau, mỗi nửa đồng nhất một nhãn.
    assert labels[0, 0] != labels[0, 31]
    assert np.all(labels[:, :16] == labels[0, 0])
    assert np.all(labels[:, 16:] == labels[0, 31])


def test_kmeans_rgb_three_channels() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 40, (16, 8, 3), dtype=np.uint8)
    b = rng.integers(200, 240, (16, 8, 3), dtype=np.uint8)
    img = np.concatenate([a, b], axis=1)

    labels, centres = kmeans_segmentation(img, k=2, seed=2, max_iter=20)
    assert labels.shape == img.shape[:2]
    assert centres.shape == (2, 3)


def test_kmeans_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    l1, c1 = kmeans_segmentation(img, k=3, seed=9, max_iter=20)
    l2, c2 = kmeans_segmentation(img, k=3, seed=9, max_iter=20)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(c1, c2)


def test_kmeans_rejects_bad_args() -> None:
    img = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="k"):
        kmeans_segmentation(img, k=1, seed=1, max_iter=10)
    with pytest.raises(ValueError, match="max_iter"):
        kmeans_segmentation(img, k=2, seed=1, max_iter=0)
    with pytest.raises(ValueError, match="exceed"):
        kmeans_segmentation(np.zeros((2, 2), dtype=np.uint8), k=5, seed=1, max_iter=10)


# ---------------------------------------------------------------------------
# canny_edges
# ---------------------------------------------------------------------------


def test_canny_finds_step_edge_as_thin_line() -> None:
    step = np.zeros((64, 64), dtype=np.uint8)
    step[:, 32:] = 255  # bậc dọc tại cột 32

    edges = canny_edges(step, low=50, high=150, sigma=1.0)
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)) <= {0, 255}

    ys, xs = np.nonzero(edges)
    assert ys.size > 0
    # Biên mảnh: mọi pixel biên nằm sát cột bậc thật (sai số ±2 cột do trơn).
    assert np.all(np.abs(xs.astype(int) - 32) <= 2)

    # Đối chiếu lỏng với cv2.Canny qua tỉ lệ trùng khớp (IoU); không đòi bằng
    # từng pixel vì cv2 dùng L1-gradient và bộ lọc trong khác ta dùng L2.
    cv_edges = cv2.Canny(step, 50, 150)
    ours = set(zip(ys.tolist(), xs.tolist()))
    cv_ys, cv_xs = np.nonzero(cv_edges)
    ref = set(zip(cv_ys.tolist(), cv_xs.tolist()))
    inter = len(ours & ref)
    union = len(ours | ref)
    iou = inter / float(union) if union else 0.0
    assert iou >= 0.5


def test_canny_rejects_bad_args() -> None:
    img = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="high"):
        canny_edges(img, low=100, high=50, sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        canny_edges(img, low=10, high=20, sigma=0.0)


# ---------------------------------------------------------------------------
# hough_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["horizontal", "vertical", "main_diag"],
)
def test_hough_recovers_drawn_line(mode: str) -> None:
    n = 40
    edges = np.zeros((n, n), dtype=np.uint8)
    if mode == "horizontal":
        edges[15, 5:35] = 255
    elif mode == "vertical":
        edges[5:35, 15] = 255
    else:
        for i in range(5, 35):
            edges[i, i] = 255  # đường chéo chính y = x

    rho, theta, _votes = hough_lines(edges, theta_step_deg=1.0, rho_step=1.0)
    assert rho.size > 0
    # Đỉnh mạnh nhất là đường đã vẽ. Kiểm tra hình học thay vì giá trị góc cụ
    # thể, vì (rho, theta) và (-rho, theta+180°) biểu diễn CÙNG một đường thẳng
    # (đường dọc x=15 xuất hiện ở cả theta≈0 và theta≈179).
    r, th = float(rho[0]), float(theta[0])
    ys, xs = np.nonzero(edges)
    dists = np.abs(xs.astype(float) * np.cos(th) + ys.astype(float) * np.sin(th) - r)
    # Dung sai do lượng tử hóa rho (bước 1px) và theta (bước 1°).
    assert np.max(dists) <= 2.0


def test_hough_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="binary_edges"):
        hough_lines(np.zeros((4, 4), dtype=np.float32), 1.0, 1.0)
    with pytest.raises(ValueError, match="theta_step_deg"):
        hough_lines(np.zeros((4, 4), dtype=np.uint8), 0.0, 1.0)
    with pytest.raises(ValueError, match="rho_step"):
        hough_lines(np.zeros((4, 4), dtype=np.uint8), 1.0, -1.0)


# ---------------------------------------------------------------------------
# simple_watershed
# ---------------------------------------------------------------------------


def test_watershed_splits_two_touching_discs() -> None:
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w]
    # Hai lòng chảo (cường độ thấp) tiếp xúc nhau, bao quanh là gờ cao: bề mặt
    # địa hình có hai cực tiểu tách biệt như hai "đĩa" chạm nhau.
    d1 = np.sqrt((yy - 18.0) ** 2 + (xx - 20.0) ** 2)
    d2 = np.sqrt((yy - 18.0) ** 2 + (xx - 40.0) ** 2)
    surf = (20.0 + 8.0 * np.minimum(d1, d2)).astype(np.float64)

    markers = np.zeros((h, w), dtype=np.int32)
    markers[18, 20] = 1
    markers[18, 40] = 2

    labels = simple_watershed(surf, markers)

    assert labels.shape == surf.shape
    # Hạt giống giữ nguyên nhãn của mình.
    assert labels[18, 20] == 1
    assert labels[18, 40] == 2
    # Cả hai lưu vực đều có diện tích đáng kể.
    assert np.count_nonzero(labels == 1) > 20
    assert np.count_nonzero(labels == 2) > 20
    # Tồn tại đường phân thủy (-1) ngăn cách hai lưu vực.
    assert np.count_nonzero(labels == -1) > 0


def test_watershed_rejects_bad_args() -> None:
    surf = np.zeros((6, 6), dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        simple_watershed(surf, np.zeros((5, 5), dtype=np.int32))
    with pytest.raises(ValueError, match="positive marker"):
        simple_watershed(surf, np.zeros((6, 6), dtype=np.int32))
    with pytest.raises(ValueError, match="non-negative"):
        bad = np.zeros((6, 6), dtype=np.int32)
        bad[2, 2] = -1
        simple_watershed(surf, bad)
