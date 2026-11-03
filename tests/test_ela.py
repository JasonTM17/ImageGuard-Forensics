"""ELA: applicability gate + thống kê thứ tự (không ngưỡng cứng)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA


def test_png_not_applicable(valid_png: Path, ts) -> None:
    doc = safe_open(valid_png, ts)
    assert isinstance(doc, ImageDocument)
    r = ELA().analyze(doc, ts)
    assert r.verdict == Verdict.NOT_APPLICABLE
    assert any("ELA phân tích JPEG" in lim for lim in r.limitations)


def test_jpeg_runs_and_returns_heatmap(valid_jpg: Path, ts) -> None:
    doc = safe_open(valid_jpg, ts)
    assert isinstance(doc, ImageDocument)
    r = ELA().analyze(doc, ts)
    assert r.verdict != Verdict.NOT_APPLICABLE
    assert r.heatmap is not None and 0 <= r.heatmap.min() <= r.heatmap.max() <= 1
    assert r.score is not None and r.score_definition  # integrity: score+def


def test_tampered_region_higher_ela(tmp_path: Path, ts) -> None:
    """Ảnh lưu JPEG lần 1, sau đó dán vùng từ ảnh khác rồi lưu lại → vùng dán
    có lịch sử nén khác → ELA cao hơn. Assert THỨ TỰ thống kê, không ngưỡng."""
    rng = np.random.default_rng(4)
    base = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    p1 = tmp_path / "first.jpg"
    cv2.imwrite(str(p1), base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img = cv2.imread(str(p1))
    # Dán một vùng ảnh khác (chưa qua JPEG q95) vào giữa.
    patch = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    img[100:180, 100:180] = patch
    p2 = tmp_path / "second.jpg"
    cv2.imwrite(str(p2), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    doc = safe_open(p2, ts)
    assert isinstance(doc, ImageDocument)
    r = ELA().analyze(doc, ts)
    assert r.heatmap is not None
    inside = r.heatmap[110:170, 110:170].mean()
    outside = np.concatenate(
        [r.heatmap[0:80].ravel(), r.heatmap[220:].ravel()]
    ).mean()
    assert inside > outside  # thứ tự: vùng dán phải nóng hơn nền
