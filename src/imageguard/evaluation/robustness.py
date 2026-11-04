"""Robustness battery — đo detector chịu được distortion nào.

Với mỗi forged sample: áp biến đổi (JPEG q, rotate, scale) rồi chạy lại
method; ghi detection rate per condition. Dùng cho L3-prep (so sánh
traditional vs DL sau này) và capstone report — số đo THẬT, không giả định.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.forensics.base import Method

_DISTORTIONS = {
    "none": lambda img: img,
    "jpeg_90": lambda img: _jpeg(img, 90),
    "jpeg_70": lambda img: _jpeg(img, 70),
    "rotate_5": lambda img: _rotate(img, 5),
    "scale_0.9": lambda img: _scale(img, 0.9),
}


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _scale(img, s):
    return cv2.resize(img, None, fx=s, fy=s)


def run_battery(
    forged_paths: list[Path],
    methods: list[Method],
    cfg: ThresholdSet,
    distortions: dict | None = None,
) -> dict:
    """Trả dict {method: {distortion: {n, n_suspicious, n_inconclusive, rate}}}."""
    distortions = _DISTORTIONS if distortions is None else distortions
    out: dict[str, dict] = {m.name: {} for m in methods}
    for m in methods:
        for dname, fn in distortions.items():
            n = n_sus = n_inc = 0
            for p in forged_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                warped = fn(img)
                # Chạy trực tiếp trên ImageDocument-lite: tái dùng safe_open
                # bằng cách encode lại — giữ đúng contract method.
                ok, buf = cv2.imencode(".png", warped)
                if not ok:
                    continue
                tmp = p.parent / f".__rob_{dname}.png"
                tmp.write_bytes(buf.tobytes())
                doc = safe_open(tmp, cfg)
                tmp.unlink(missing_ok=True)
                if hasattr(doc, "code"):
                    continue
                r = m.analyze(doc, cfg)
                n += 1
                if r.verdict == Verdict.SUSPICIOUS_INDICATION:
                    n_sus += 1
                elif r.verdict == Verdict.INCONCLUSIVE:
                    n_inc += 1
            out[m.name][dname] = {
                "n": n, "n_suspicious": n_sus, "n_inconclusive": n_inc,
                "detection_rate": round(n_sus / n, 3) if n else None,
            }
    return out


def save_battery(res: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return path
