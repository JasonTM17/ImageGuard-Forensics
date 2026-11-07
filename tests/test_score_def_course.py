"""Course lock: a score without a definition names the method."""

from __future__ import annotations

import pytest

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult


def test_score_without_definition_names_the_method() -> None:
    with pytest.raises(ValueError) as caught:
        AlgorithmResult(
            method=MethodName.COPY_MOVE_KEYPOINT,
            verdict=Verdict.SUSPICIOUS_INDICATION,
            score=5.0,
            score_definition="",
        )
    assert str(caught.value) == (
        "MethodName.COPY_MOVE_KEYPOINT: score=5.0 thiếu score_definition"
    )
