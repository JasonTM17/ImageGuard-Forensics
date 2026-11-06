"""Course lock: ORB backend also convicts the textured translate forge.

Same seed-23 384×384 translation paste as the keypoint course. With
``keypoint.backend=orb`` and default thresholds otherwise, KeypointCopyMove
returns SUSPICIOUS_INDICATION, reports backend orb, and the mask touches both
source and destination rectangles.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove

_SIZE = 384
_DX, _DY = 48, 36
_SX, _SY, _RW, _RH = 80, 80, 140, 140


def _doc_from(bgr: np.ndarray) -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="o" * 64,
        file_bytes=0,
        file_name="orb.png",
    )


def _orb_thresholds() -> ThresholdSet:
    ts = load_thresholds()
    raw = json.loads(json.dumps(ts.raw))
    raw["keypoint"]["backend"] = {"value": "orb", "source": "test"}
    return ThresholdSet(raw=raw, config_hash=ts.config_hash)


def _textured_base(seed: int = 23, size: int = _SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = h = size
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    img = np.tile(x, (h, 1)) * 0.5 + np.tile(y, (w, 1)).T * 0.5
    img = cv2.merge([img, np.roll(img, 32, axis=1), np.roll(img, 16, axis=0)])
    for _ in range(int(rng.integers(25, 45))):
        c = tuple(int(v) for v in rng.integers(30, 225, 3))
        if rng.random() < 0.5:
            x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
            x1 = min(w, x0 + int(rng.integers(8, 80)))
            y1 = min(h, y0 + int(rng.integers(8, 80)))
            cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
        else:
            cx = int(rng.integers(20, w - 20))
            cy = int(rng.integers(20, h - 20))
            cv2.circle(img, (cx, cy), int(rng.integers(5, 50)), c, -1)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    for _ in range(int(w * h * 0.006)):
        cx = int(rng.integers(2, w - 2))
        cy = int(rng.integers(2, h - 2))
        r = int(rng.integers(1, 3))
        cv2.circle(
            img,
            (cx, cy),
            r,
            tuple(int(v) for v in rng.integers(0, 255, 3)),
            -1,
        )
    noise = rng.normal(0, 18, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    img += noise[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_orb_backend_translate_copy_move_suspicious() -> None:
    base = _textured_base()
    forged = base.copy()
    forged[
        _SY + _DY : _SY + _DY + _RH,
        _SX + _DX : _SX + _DX + _RW,
    ] = base[_SY : _SY + _RH, _SX : _SX + _RW]
    result = KeypointCopyMove().analyze(_doc_from(forged), _orb_thresholds())

    assert result.verdict == Verdict.SUSPICIOUS_INDICATION
    assert result.evidence.get("backend") == "orb"
    assert result.mask is not None
    assert np.any(result.mask[_SY : _SY + _RH, _SX : _SX + _RW] > 0)
    assert np.any(
        result.mask[
            _SY + _DY : _SY + _DY + _RH,
            _SX + _DX : _SX + _DX + _RW,
        ]
        > 0
    )
