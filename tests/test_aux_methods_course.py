"""Course lock: AUX_METHODS stays the six exploratory cue names."""

from __future__ import annotations

from imageguard.forensics.fusion import AUX_METHODS

_LOCKED = [
    "ela",
    "exif_audit",
    "frequency_analysis",
    "noise_inconsistency",
    "noise_local",
    "splicing_cues",
]


def test_aux_methods_values_locked() -> None:
    assert sorted(method.value for method in AUX_METHODS) == _LOCKED
