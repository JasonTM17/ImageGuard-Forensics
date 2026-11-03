"""Method registry — spec string → Method instance. Dùng chung CLI + service."""

from __future__ import annotations

import importlib

from imageguard.forensics.base import Method

METHOD_SPECS = {
    "keypoint": "imageguard.forensics.keypoint.pipeline:KeypointCopyMove",
    "block": "imageguard.forensics.block.pipeline:BlockCopyMove",
    "ela": "imageguard.forensics.auxiliary.ela:ELA",
    "noise": "imageguard.forensics.auxiliary.noise_map:NoiseMap",
    "noise_local": "imageguard.forensics.auxiliary.noise_local:NoiseLocal",
    "freq": "imageguard.forensics.auxiliary.freq_view:FreqView",
    "splicing": "imageguard.forensics.auxiliary.splicing:SplicingCues",
}


def load_method(spec: str) -> Method:
    path = METHOD_SPECS.get(spec)
    if path is None:
        raise KeyError(f"method '{spec}' không có trong registry: {list(METHOD_SPECS)}")
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)()
