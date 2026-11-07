"""Course lock: report export drops arrays and sums method durations."""

from __future__ import annotations

import numpy as np

from imageguard.core.enums import MethodName, Verdict
from imageguard.core.models import AlgorithmResult, ForensicReport
from imageguard.services.analysis_service import report_dur_ms, report_to_dict


def test_report_dict_drops_arrays_and_sums_duration() -> None:
    report = ForensicReport(
        report_id="r",
        image_sha256="ab" * 32,
        image_name="a.png",
        image_width=8,
        image_height=8,
        created_at="t",
        tool_version="0",
        methods=[
            AlgorithmResult(
                method=MethodName.ELA,
                verdict=Verdict.NO_INDICATION,
                duration_ms=3,
                evidence={"mask": np.zeros((2, 2), np.uint8), "n": 1},
            ),
            AlgorithmResult(
                method=MethodName.EXIF_AUDIT,
                verdict=Verdict.INCONCLUSIVE,
                duration_ms=4,
            ),
        ],
    )
    exported = report_to_dict(report)
    assert report_dur_ms(report) == 7
    assert exported["methods"][0]["evidence"] == {"n": 1}
    assert exported["methods"][0]["has_mask"] is False
    assert exported["methods"][0]["duration_ms"] == 3
    assert exported["methods"][1]["evidence"] == {}
    assert exported["methods"][1]["duration_ms"] == 4
    assert exported["overall_verdict"] == "inconclusive"