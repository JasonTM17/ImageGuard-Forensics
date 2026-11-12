"""Script trình diễn toolkit ``imageguard.ipkit`` theo chương C2–C7 (HCMUTE, dunghv).

Mỗi chương ghi một ảnh PNG montage (trái: gốc | phải: kết quả) bằng OpenCV chỉ
để đọc/ghi file và vẽ chú thích; mọi toán tử xử lý ảnh gọi từ ipkit (NumPy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Đưa ``src`` vào PYTHONPATH khi chạy trực tiếp ``python scripts/ipkit_demo.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from imageguard.ipkit import (
    classifiers,
    features_basic,
    features_hog,
    frequency,
    histogram,
    intensity,
    morph_algos,
    segmentation,
    spatial,
    structuring_element,
    threshold,
)

# Seed cố định để ảnh tổng hợp và C7 lặp lại được khi báo cáo.
DEMO_SEED = 20261112
# Kích thước nhỏ để demo pytest chạy nhanh; bội của 8 cho HOG (ô cell).
DEMO_H = 128
DEMO_W = 160

# Tên file đầu ra theo chương — test kiểm tra danh sách này.
CHAPTER_OUTPUT_NAMES: tuple[str, ...] = (
    "c2_montage.png",
    "c3_montage.png",
    "c4_montage.png",
    "c5_montage.png",
    "c6_montage.png",
    "c7_montage.png",
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAPTION_H = 22
_GAP = 4


def validate_out_dir(path: Path) -> Path:
    """Tạo thư mục đầu ra và trả về đường dẫn đã resolve.

    Parameters
    ----------
    path:
        Thư mục do người dùng chỉ định (--out).

    Returns
    -------
    Path
        Thư mục tồn tại, dùng làm gốc cho mọi file PNG.

    Raises
    ------
    ValueError
        Nếu không tạo được thư mục.
    """
    resolved = path.expanduser().resolve()
    # mkdir parents: người dùng có thể truyền đường dẫn chưa có.
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError(f"output path is not a directory: {resolved}")
    return resolved


def resolve_output_file(out_dir: Path, filename: str) -> Path:
    """Ghép ``filename`` vào ``out_dir`` và chặn path traversal ra ngoài thư mục đích.

    Parameters
    ----------
    out_dir:
        Thư mục gốc đã validate.
    filename:
        Tên file đơn giản (không chứa ``..``).

    Returns
    -------
    Path
        Đường dẫn an toàn để ghi PNG.

    Raises
    ------
    ValueError
        Nếu tên file hoặc đường dẫn resolve trượt ra ngoài ``out_dir``.
    """
    if Path(filename).name != filename or ".." in Path(filename).parts:
        raise ValueError(f"unsafe output filename: {filename!r}")
    target = (out_dir / filename).resolve()
    base = out_dir.resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"refusing to write outside output directory: {target}")
    return target


def synthetic_demo_image(seed: int = DEMO_SEED) -> np.ndarray:
    """Sinh ảnh BGR uint8 tất định phục vụ trình diễn (không đọc file ngoài).

    Ảnh gồm nền gradient, khối sáng, đường thẳng tối (Hough/Canny), vùng màu
    (K-means) và chấm nhiễu nhỏ (lọc không gian).

    Parameters
    ----------
    seed:
        Seed cho ``numpy.random.Generator``.

    Returns
    -------
    np.ndarray
        Ảnh màu BGR ``(DEMO_H, DEMO_W, 3)`` kiểu ``uint8``.

    Example
    -------
    >>> img = synthetic_demo_image(1)
    >>> img.shape == (DEMO_H, DEMO_W, 3) and img.dtype == np.uint8
    True
    """
    rng = np.random.default_rng(seed)
    h, w = DEMO_H, DEMO_W
    ys = np.linspace(35, 190, h, dtype=np.float64)[:, None]
    xs = np.linspace(25, 210, w, dtype=np.float64)[None, :]
    # Nền xám gradient: mô phỏng vùng tối/sáng cho log/gamma/histogram.
    gray = np.clip((ys + xs) * 0.55, 0, 255).astype(np.uint8)
    gray[28:72, 38:102] = 225
    gray[84:86, 18:138] = 25
    for _ in range(6):
        cy = int(rng.integers(8, h - 8))
        cx = int(rng.integers(8, w - 8))
        rad = int(rng.integers(2, 5))
        gray[cy - rad : cy + rad + 1, cx - rad : cx + rad + 1] = int(
            rng.integers(0, 256)
        )
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Vùng xanh cho phân cụm màu (C5).
    bgr[48:58, 98:118] = (40, 210, 30)
    return bgr


def gray_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Chuyển BGR sang xám uint8 (chỉ dùng OpenCV cho đổi không gian màu)."""
    if bgr.ndim == 2:
        return np.asarray(bgr, dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_bgr_display(img: np.ndarray) -> np.ndarray:
    """Chuẩn hoá ảnh 2D bool/int/gray sang BGR để ghép montage."""
    arr = np.asarray(img)
    if arr.dtype == bool:
        out = np.where(arr, 255, 0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 2:
        if arr.dtype in (np.int32, np.int64):
            # Nhãn thành phần: tô màu giả để trình chiếu.
            return colorize_labels(arr)
        return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr.astype(np.uint8))
    raise ValueError(f"cannot display array shape {arr.shape}")


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Tô màu bản đồ nhãn nguyên (0 = đen, khác 0 = màu khác nhau)."""
    lab = np.asarray(labels, dtype=np.int32)
    h, w = lab.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    max_lab = int(lab.max())
    if max_lab == 0:
        return bgr
    rng = np.random.default_rng(0)
    palette = rng.integers(32, 255, size=(max_lab + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    for idx in range(1, max_lab + 1):
        bgr[lab == idx] = palette[idx]
    return bgr


def colorize_kmeans_labels(labels: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Hiển thị nhãn K-means: mỗi cụm dùng màu trung tâm cụm (gần ảnh gốc)."""
    h, w = labels.shape
    if centres.shape[1] == 1:
        out = np.zeros((h, w), dtype=np.uint8)
        for k in range(centres.shape[0]):
            out[labels == k] = int(np.clip(np.rint(centres[k, 0]), 0, 255))
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(centres.shape[0]):
        c = np.clip(np.rint(centres[k]), 0, 255).astype(np.uint8)
        if c.size == 3:
            bgr = (int(c[0]), int(c[1]), int(c[2]))
        else:
            bgr = (int(c[0]), int(c[0]), int(c[0]))
        out[labels == k] = bgr
    return out


def build_pair_row(
    left: np.ndarray,
    right: np.ndarray,
    caption: str,
) -> np.ndarray:
    """Một hàng montage: thanh chú thích + cặp ảnh trái/phải."""
    left_b = to_bgr_display(left)
    right_b = to_bgr_display(right)
    # Căn cùng chiều cao để hàng thẳng hàng khi ghép dọc.
    target_h = max(left_b.shape[0], right_b.shape[0])
    if left_b.shape[0] != target_h:
        left_b = cv2.resize(
            left_b, (left_b.shape[1], target_h), interpolation=cv2.INTER_NEAREST
        )
    if right_b.shape[0] != target_h:
        right_b = cv2.resize(
            right_b, (right_b.shape[1], target_h), interpolation=cv2.INTER_NEAREST
        )
    sep = np.full((target_h, _GAP, 3), 40, dtype=np.uint8)
    pair = np.hstack([left_b, sep, right_b])
    cap_bar = np.full((_CAPTION_H, pair.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        cap_bar,
        caption,
        (4, 16),
        _FONT,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([cap_bar, pair])


def stack_montage(rows: list[np.ndarray]) -> np.ndarray:
    """Ghép các hàng montage theo chiều dọc, căn cùng chiều rộng."""
    if not rows:
        raise ValueError("montage needs at least one row")
    max_w = max(r.shape[1] for r in rows)
    padded: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 20, dtype=np.uint8)
            padded.append(np.hstack([row, pad]))
        else:
            padded.append(row)
    return np.vstack(padded)


def write_png_safe(out_dir: Path, filename: str, bgr: np.ndarray) -> Path:
    """Ghi PNG chỉ khi đường dẫn nằm trong ``out_dir``."""
    path = resolve_output_file(out_dir, filename)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("write_png_safe expects BGR HxWx3")
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise OSError(f"cv2.imwrite failed: {path}")
    return path


def demo_chapter_c2(out_dir: Path, bgr: np.ndarray) -> Path:
    """Chương 2: cường độ, histogram và lọc không gian (mean/Gaussian/median/unsharp).

    Parameters
    ----------
    out_dir:
        Thư mục đầu ra đã validate.
    bgr:
        Ảnh đầu vào BGR.

    Returns
    -------
    Path
        Đường dẫn ``c2_montage.png``.
    """
    gray = gray_from_bgr(bgr)
    rows: list[np.ndarray] = []
    # --- Cường độ (Ch.2 mục biến đổi điểm) ---
    rows.append(build_pair_row(bgr, intensity.negative(bgr), "C2: Am ban (s = L-1-r)"))
    rows.append(build_pair_row(bgr, intensity.log_transform(bgr), "C2: Bien doi log"))
    rows.append(
        build_pair_row(bgr, intensity.gamma_transform(bgr, 1.8), "C2: Gamma y=1.8")
    )
    rows.append(
        build_pair_row(
            gray,
            histogram.equalize(gray),
            "C2: Can bang histogram (CDF)",
        )
    )
    # --- Lọc không gian (Ch.2 smoothing / unsharp) ---
    rows.append(
        build_pair_row(bgr, spatial.mean_filter(bgr, 5), "C2: Loc trung binh 5x5")
    )
    rows.append(
        build_pair_row(bgr, spatial.gaussian_filter(bgr, 5, 1.0), "C2: Loc Gaussian")
    )
    rows.append(build_pair_row(bgr, spatial.median_filter(bgr, 5), "C2: Loc trung vi"))
    rows.append(
        build_pair_row(
            bgr,
            spatial.unsharp_mask(bgr, 5, 1.0, 1.0),
            "C2: Unsharp masking",
        )
    )
    montage = stack_montage(rows)
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[0], montage)


def demo_chapter_c3(out_dir: Path, bgr: np.ndarray) -> Path:
    """Chương 3: lọc tần số ideal/Butterworth/Gaussian (LP+HP) và phổ biên độ."""
    gray = gray_from_bgr(bgr)
    spec = frequency.magnitude_spectrum(gray)
    M, N = gray.shape
    pad_shape = (2 * M, 2 * N)
    d0 = 28.0
    order = 2
    filters: list[tuple[str, np.ndarray]] = [
        (
            "Ideal LP",
            frequency.apply_frequency_filter(
                gray, frequency.ideal_lowpass(pad_shape, d0)
            ),
        ),
        (
            "Ideal HP",
            frequency.apply_frequency_filter(
                gray, frequency.ideal_highpass(pad_shape, d0)
            ),
        ),
        (
            "Butter LP",
            frequency.apply_frequency_filter(
                gray, frequency.butterworth_lowpass(pad_shape, d0, order)
            ),
        ),
        (
            "Butter HP",
            frequency.apply_frequency_filter(
                gray, frequency.butterworth_highpass(pad_shape, d0, order)
            ),
        ),
        (
            "Gauss LP",
            frequency.apply_frequency_filter(
                gray, frequency.gaussian_lowpass(pad_shape, d0)
            ),
        ),
        (
            "Gauss HP",
            frequency.apply_frequency_filter(
                gray, frequency.gaussian_highpass(pad_shape, d0)
            ),
        ),
    ]
    rows = [
        build_pair_row(
            gray,
            spec,
            "C3: Anh goc | pho bien do log (DFT)",
        )
    ]
    for title, filtered in filters:
        rows.append(build_pair_row(gray, filtered, f"C3: {title} d0={d0:.0f}"))
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[1], stack_montage(rows))


def demo_chapter_c4(out_dir: Path, bgr: np.ndarray) -> Path:
    """Chương 4: biên, xương, thành phần liên thông trên ảnh đã ngưỡng Otsu."""
    gray = gray_from_bgr(bgr)
    t = threshold.otsu_threshold(gray)
    binary = intensity.threshold_binary(gray, t)
    se = structuring_element("rect", 3)
    boundary = morph_algos.boundary_extraction(binary, se)
    skel = morph_algos.skeleton(binary, se)
    labels, _ = morph_algos.connected_components(binary, 4)
    rows = [
        build_pair_row(gray, binary, f"C4: Nguong Otsu T={t}"),
        build_pair_row(binary, boundary, "C4: Trich bien beta(A)"),
        build_pair_row(binary, skel, "C4: Xuong Lantuejoul"),
        build_pair_row(binary, labels, "C4: Thanh phan lien thong (mau)"),
    ]
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[2], stack_montage(rows))


def _draw_hough_overlay(edges: np.ndarray, rhos, thetas, votes) -> np.ndarray:
    """Vẽ vài đường Hough mạnh nhất lên ảnh biên (OpenCV chỉ để minh họa)."""
    canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    n_draw = min(3, len(rhos))
    for i in range(n_draw):
        rho = float(rhos[i])
        theta = float(thetas[i])
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    if n_draw:
        cv2.putText(
            canvas,
            f"top votes={int(votes[0])}",
            (4, 14),
            _FONT,
            0.4,
            (200, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def demo_chapter_c5(out_dir: Path, bgr: np.ndarray) -> Path:
    """Chương 5: Canny, Otsu, K-means, biến đổi Hough."""
    gray = gray_from_bgr(bgr)
    edges = segmentation.canny_edges(gray, low=40.0, high=120.0, sigma=1.2)
    t = threshold.otsu_threshold(gray)
    otsu_bin = intensity.threshold_binary(gray, t)
    labels, centres = segmentation.kmeans_segmentation(
        bgr, k=3, seed=DEMO_SEED, max_iter=15
    )
    kmeans_vis = colorize_kmeans_labels(labels, centres)
    rhos, thetas, votes = segmentation.hough_lines(
        edges, theta_step_deg=2.0, rho_step=1.0
    )
    hough_vis = _draw_hough_overlay(edges, rhos, thetas, votes)
    rows = [
        build_pair_row(gray, edges, "C5: Bien Canny"),
        build_pair_row(gray, otsu_bin, f"C5: Phan doan Otsu T={t}"),
        build_pair_row(bgr, kmeans_vis, "C5: K-means k=3"),
        build_pair_row(edges, hough_vis, "C5: Hough lines (top peaks)"),
    ]
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[3], stack_montage(rows))


def demo_chapter_c6(out_dir: Path, bgr: np.ndarray) -> Path:
    """Chương 6: LBP và trực quan hoá HOG."""
    gray = gray_from_bgr(bgr)
    lbp = features_basic.lbp_image(gray, radius=1, neighbors=8)
    # Nhân 17 để mã LBP dễ nhìn hơn khi trình chiếu (chỉ hiển thị).
    lbp_vis = np.clip(lbp.astype(np.uint16) * 17, 0, 255).astype(np.uint8)
    hog_vis = features_hog.hog_visualize(gray, cell=8, bins=9, signed=False)
    rows = [
        build_pair_row(gray, lbp_vis, "C6: Ma LBP (8 lang gieng)"),
        build_pair_row(gray, hog_vis, "C6: HOG visualize (cell=8)"),
    ]
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[4], stack_montage(rows))


def synthetic_blob_features(seed: int = DEMO_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Tạo đặc trưng 2D và nhãn hai lớp blob tổng hợp cho demo phân lớp C7."""
    rng = np.random.default_rng(seed + 7)
    n_per = 24
    xs: list[list[float]] = []
    ys: list[int] = []
    # Lớp 0: blob nhỏ, tối; lớp 1: blob lớn, sáng — phân tách bằng mean distance.
    for label, (mu_r, mu_i) in enumerate([(4.0, 70.0), (11.0, 175.0)]):
        for _ in range(n_per):
            radius = float(rng.normal(mu_r, 0.8))
            intensity = float(rng.normal(mu_i, 12.0))
            xs.append([radius * radius, intensity])
            ys.append(label)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def demo_classifier_stdout(seed: int = DEMO_SEED) -> str:
    """In ma trận nhầm lẫn của minimum-distance classifier trên blob tổng hợp.

    Returns
      -------
      str
          Nội dung đã in (để test có thể đọc lại).
    """
    X, y = synthetic_blob_features(seed)
    rng = np.random.default_rng(seed + 99)
    order = rng.permutation(len(y))
    split = (2 * len(y)) // 3
    tr, te = order[:split], order[split:]
    y_pred = classifiers.minimum_distance_classifier(X[tr], y[tr], X[te])
    cm = classifiers.confusion_matrix(y[te], y_pred)
    acc = classifiers.accuracy_score(y[te], y_pred)
    lines = [
        "=== C7: Phan lop minimum distance (blob tong hop) ===",
        f"accuracy (tap test): {acc:.4f}",
        "confusion_matrix (hang=that, cot=du doan):",
        np.array2string(cm),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    return text


def demo_chapter_c7(out_dir: Path, seed: int = DEMO_SEED) -> Path:
    """Chương 7: montage minh họa hai lớp blob và gọi in confusion matrix."""
    demo_classifier_stdout(seed)
    rng = np.random.default_rng(seed + 3)
    tile = 48
    left = np.full((tile, tile, 3), 50, dtype=np.uint8)
    right = np.full((tile, tile, 3), 200, dtype=np.uint8)
    for _ in range(8):
        cy, cx = int(rng.integers(6, tile - 6)), int(rng.integers(6, tile - 6))
        left[cy - 2 : cy + 3, cx - 2 : cx + 3] = (90, 90, 90)
    for _ in range(8):
        cy, cx = int(rng.integers(8, tile - 8)), int(rng.integers(8, tile - 8))
        right[cy - 5 : cy + 6, cx - 5 : cx + 6] = (240, 240, 240)
    rows = [
        build_pair_row(
            left, right, "C7: Lop 0 (blob nho) | Lop 1 (blob lon) — tong hop"
        ),
    ]
    return write_png_safe(out_dir, CHAPTER_OUTPUT_NAMES[5], stack_montage(rows))


def load_input_image(path: Path | None) -> np.ndarray:
    """Đọc ảnh người dùng hoặc sinh ảnh tổng hợp."""
    if path is None:
        return synthetic_demo_image(DEMO_SEED)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"could not decode image: {path}")
    # Thu nhỏ nếu quá lớn để demo không chậm khi trình chiếu.
    h, w = bgr.shape[:2]
    scale = min(1.0, DEMO_H / h, DEMO_W / w)
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def run_all_demos(out_dir: Path, bgr: np.ndarray, seed: int = DEMO_SEED) -> list[Path]:
    """Chạy lần lượt demo C2–C7 và trả về danh sách file đã ghi."""
    written: list[Path] = []
    written.append(demo_chapter_c2(out_dir, bgr))
    written.append(demo_chapter_c3(out_dir, bgr))
    written.append(demo_chapter_c4(out_dir, bgr))
    written.append(demo_chapter_c5(out_dir, bgr))
    written.append(demo_chapter_c6(out_dir, bgr))
    written.append(demo_chapter_c7(out_dir, seed))
    return written


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI: ``--out`` bắt buộc, ``--image`` tùy chọn.

    Returns
    -------
    int
        0 khi thành công; mã lỗi khác 0 khi thất bại.
    """
    parser = argparse.ArgumentParser(
        description="Trinh dien imageguard.ipkit theo chuong C2-C7 (HCMUTE).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Thu muc ghi PNG montage (duoc tao neu chua co).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Duong dan anh BGR; mac dinh dung anh tong hop seeded.",
    )
    args = parser.parse_args(argv)
    try:
        out_dir = validate_out_dir(args.out)
        bgr = load_input_image(args.image)
        paths = run_all_demos(out_dir, bgr)
        for p in paths:
            print(f"wrote {p}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
