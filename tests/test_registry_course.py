"""Course lock: METHOD_SPECS keys and load_method wiring stay frozen."""

from __future__ import annotations

import pytest

from imageguard.forensics.registry import METHOD_SPECS, load_method

_LOCKED_SPECS = [
    "block",
    "ela",
    "freq",
    "keypoint",
    "noise",
    "noise_local",
    "splicing",
]


def test_method_specs_and_loaders_locked() -> None:
    assert sorted(METHOD_SPECS) == _LOCKED_SPECS
    for key in _LOCKED_SPECS:
        method = load_method(key)
        assert callable(getattr(method, "analyze", None))
    with pytest.raises(KeyError):
        load_method("nope")
