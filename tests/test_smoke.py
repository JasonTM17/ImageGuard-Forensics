"""Smoke tests — package import được và môi trường có đủ primitive cốt lõi."""


def test_package_import() -> None:
    import imageguard

    assert imageguard.__version__


def test_opencv_sift_available() -> None:
    # SIFT thuộc opencv-python chính từ 4.4 (patent hết hạn 2020) — assert runtime,
    # không giả định. Nếu fail → ORB làm backend chính (xem configs/thresholds.yaml).
    import cv2

    assert hasattr(cv2, "SIFT_create")


def test_numpy_opencv_versions() -> None:
    import cv2
    import numpy as np

    assert np.__version__ and cv2.__version__
