"""Course lock: load_method rejects unknown registry keys with KeyError."""

from __future__ import annotations

import pytest

from imageguard.forensics.registry import METHOD_SPECS, load_method


def test_unknown_method_key_raises_keyerror_naming_known_specs() -> None:
    with pytest.raises(KeyError) as err:
        load_method("not_a_real_method")
    message = str(err.value)
    assert "not_a_real_method" in message
    for key in ("block", "ela", "keypoint"):
        assert key in message
    assert "not_a_real_method" not in METHOD_SPECS
