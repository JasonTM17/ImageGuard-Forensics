"""F-A4: aux methods downscale large images (aux_max_side) to bound memory.

Kernel/block sizes scale proportionally so the physical window is preserved,
and mask/heatmap are upscaled back to the original image size with the
reduction disclosed in limitations/evidence. Small images (below the cap)
must keep the exact previous full-resolution behavior.
"""

from __future__ import annotations

import json

import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.models import ImageDocument


def _cfg(cap: int) -> ThresholdSet:
    ts = load_thresholds()
    raw = json.loads(json.dumps(ts.raw))
    raw["limits"]["aux_max_side"] = {"value": cap, "source": "test"}
    return ThresholdSet(raw=raw, config_hash=ts.config_hash)


def _doc(h: int = 256, w: int = 256) -> ImageDocument:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return ImageDocument(array=img, width=w, height=h, channels=3,
                         image_format="png", sha256="x", file_bytes=1,
                         file_name="a.png")


def test_noise_map_downscales_and_upscales() -> None:
    from imageguard.forensics.auxiliary.noise_map import NoiseMap

    r = NoiseMap().analyze(_doc(), _cfg(128))
    assert r.heatmap.shape == (256, 256)
    assert r.mask is None or r.mask.shape == (256, 256)
    assert any("downscale" in l for l in r.limitations)
    assert r.evidence["scale"] == 0.5
    assert r.evidence["block"] == 8  # 16 * 0.5 — cùng kích thước vật lý


def test_noise_local_downscales_and_upscales() -> None:
    from imageguard.forensics.auxiliary.noise_local import NoiseLocal

    r = NoiseLocal().analyze(_doc(), _cfg(128))
    assert r.heatmap.shape == (256, 256)
    assert r.mask is None or r.mask.shape == (256, 256)
    assert any("downscale" in l for l in r.limitations)
    assert r.evidence["scale"] == 0.5


def test_freq_view_downscales_and_upscales() -> None:
    from imageguard.forensics.auxiliary.freq_view import FreqView

    r = FreqView().analyze(_doc(), _cfg(128))
    assert r.heatmap.shape == (256, 256)
    assert r.evidence["window"] == 4  # 8 * 0.5 — cùng kích thước vật lý
    assert any("downscale" in l for l in r.limitations)
    assert r.evidence["scale"] == 0.5


def test_small_image_keeps_full_resolution() -> None:
    from imageguard.forensics.auxiliary.noise_map import NoiseMap

    r = NoiseMap().analyze(_doc(), load_thresholds())  # 256 < aux_max_side 4096
    assert r.evidence["scale"] == 1.0
    assert not any("downscale" in l for l in r.limitations)
