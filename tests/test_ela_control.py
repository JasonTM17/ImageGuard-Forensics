"""Compare ELA heatmap contrast in controlled and pasted JPEG histories."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA


def test_ela_paste_contrast_exceeds_recompressed_control(tmp_path: Path, ts) -> None:
    rng = np.random.default_rng(23)
    base = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    q95_path = tmp_path / "base-q95.jpg"
    assert cv2.imwrite(str(q95_path), base, [cv2.IMWRITE_JPEG_QUALITY, 95])

    q95_image = cv2.imread(str(q95_path))
    assert q95_image is not None

    control_path = tmp_path / "control-q75.jpg"
    assert cv2.imwrite(
        str(control_path), q95_image, [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    pasted_image = q95_image.copy()
    patch_source = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    pasted_image[60:100, 60:100] = patch_source[60:100, 60:100]
    paste_path = tmp_path / "paste-q75.jpg"
    assert cv2.imwrite(
        str(paste_path), pasted_image, [cv2.IMWRITE_JPEG_QUALITY, 75]
    )

    control_doc = safe_open(control_path, ts)
    paste_doc = safe_open(paste_path, ts)
    assert isinstance(control_doc, ImageDocument)
    assert isinstance(paste_doc, ImageDocument)

    ela = ELA()
    control_result = ela.analyze(control_doc, ts)
    paste_result = ela.analyze(paste_doc, ts)
    assert control_result.heatmap is not None
    assert paste_result.heatmap is not None
    assert control_result.verdict != Verdict.SUSPICIOUS_INDICATION
    assert paste_result.verdict != Verdict.SUSPICIOUS_INDICATION

    outside_mask = np.ones((160, 160), dtype=bool)
    outside_mask[60:100, 60:100] = False
    control_inside_mean = float(control_result.heatmap[60:100, 60:100].mean())
    control_outside_mean = float(control_result.heatmap[outside_mask].mean())
    paste_inside_mean = float(paste_result.heatmap[60:100, 60:100].mean())
    paste_outside_mean = float(paste_result.heatmap[outside_mask].mean())

    control_delta = control_inside_mean - control_outside_mean
    paste_delta = paste_inside_mean - paste_outside_mean
    assert paste_delta > control_delta, (
        "ELA paste contrast must exceed recompressed control: "
        f"control inside={control_inside_mean:.8f}, "
        f"outside={control_outside_mean:.8f}, delta={control_delta:.8f}; "
        f"paste inside={paste_inside_mean:.8f}, "
        f"outside={paste_outside_mean:.8f}, delta={paste_delta:.8f}"
    )
