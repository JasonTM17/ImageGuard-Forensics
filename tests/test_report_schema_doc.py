"""Doc drift lock: forensic-report-schema.md khớp khóa JSON CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.services.analysis_service import (
    AnalysisRequest,
    AnalysisService,
    report_to_dict,
)

_DOC = Path(__file__).resolve().parents[1] / "docs" / "forensic-report-schema.md"
_FENCE = re.compile(
    r"```json\s+(?P<label>[\w-]+)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)


def _blocks(md: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in _FENCE.finditer(md):
        out[m.group("label")] = json.loads(m.group("body"))
    return out


def _report_key_sets(payload: dict) -> tuple[set[str], set[str], set[str]]:
    env_keys = set(payload.get("env", {}).keys())
    method_keys: set[str] = set()
    for item in payload.get("methods", []):
        method_keys |= set(item.keys())
    return set(payload.keys()), env_keys, method_keys


def _error_keys(payload: dict) -> set[str]:
    return set(payload.keys())


def _synthetic_jpeg(tmp_path: Path) -> Path:
    img = np.random.default_rng(42).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    path = tmp_path / "schema_doc.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    path.write_bytes(buf.tobytes())
    return path


def test_doc_forensic_report_contract_keys_match_report_to_dict(tmp_path: Path) -> None:
    md = _DOC.read_text(encoding="utf-8")
    blocks = _blocks(md)
    assert "forensic-report-contract" in blocks, "thiếu khối forensic-report-contract"

    svc = AnalysisService()
    image = _synthetic_jpeg(tmp_path)
    result = svc.analyze(
        AnalysisRequest(path=image, methods=["ela"], include_exif=False)
    )
    assert not isinstance(result, ErrorResult)
    live = report_to_dict(result)

    doc_top, doc_env, doc_method = _report_key_sets(blocks["forensic-report-contract"])
    live_top, live_env, live_method = _report_key_sets(live)

    assert live_top == doc_top, f"top-level: doc {doc_top ^ live_top}"
    assert live_env == doc_env, f"env: doc {doc_env ^ live_env}"
    assert live_method == doc_method, f"methods[]: doc {doc_method ^ live_method}"


def test_doc_error_result_contract_keys_match_cli_payload() -> None:
    md = _DOC.read_text(encoding="utf-8")
    blocks = _blocks(md)
    assert "error-result-contract" in blocks

    cli_shape = {
        "code": ErrorCode.DECODE_FAILED.value,
        "message": "example",
        "detail": {},
    }
    doc_keys = _error_keys(blocks["error-result-contract"])
    live_keys = _error_keys(cli_shape)
    assert live_keys == doc_keys, f"ErrorResult: doc {doc_keys ^ live_keys}"
