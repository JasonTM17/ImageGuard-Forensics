"""ReportService — read_report_json, export_markdown, and _render."""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.services.report_service import ReportService
from imageguard.storage.artifacts import ArtifactStore

_REP = {
    "overall_verdict": "suspicious_indication",
    "image_name": "a.png",
    "report_id": "r1",
    "image_width": 100,
    "image_height": 80,
    "image_sha256": "a" * 64,
    "created_at": "2024-01-01T00:00:00+00:00",
    "tool_version": "0.1.0",
    "env": {"python": "3.14", "opencv": "5.0", "numpy": "2.5"},
    "thresholds_hash": "deadbeef",
    "methods": [{
        "method": "copy_move_keypoint",
        "verdict": "suspicious_indication",
        "score": 279.0,
        "score_definition": "số cặp",
        "duration_ms": 5,
        "has_mask": True,
        "evidence": {"n_verified": 279},
        "limitations": ["mask = union"],
    }],
    "limitations": ["warn1"],
}


def test_read_report_json_missing_returns_none(tmp_path: Path) -> None:
    svc = ReportService(ArtifactStore(tmp_path / "art"))
    assert svc.read_report_json("missing-id") is None


def test_export_markdown_missing_returns_none(tmp_path: Path) -> None:
    svc = ReportService(ArtifactStore(tmp_path / "art"))
    assert svc.export_markdown("missing-id") is None


def test_render_includes_fields_and_no_probability() -> None:
    md = ReportService()._render(_REP)

    assert "# Báo cáo phân tích — a.png" in md
    assert "Có dấu hiệu nghi vấn" in md  # VERDICT_LABEL của suspicious
    assert "deadbeef" in md
    assert "n_verified: 279" in md  # evidence render dạng `k: v`
    assert "⚠ mask = union" in md
    assert "## Cảnh báo tổng" in md
    assert "warn1" in md
    # integrity: report không chứa xác suất giả mạo hay kết luận "forged"
    # (footer có câu "không chứa xác suất" — đó là tuyên bố, không phải con số)
    assert "xác suất giả mạo" not in md
    assert "forged" not in md


def test_export_markdown_writes_artifact_and_out_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")
    svc = ReportService(store)
    aid = "abc123"
    store.put_bytes(aid, "report.json", json.dumps(_REP).encode("utf-8"))

    out = tmp_path / "out.md"
    result = svc.export_markdown(aid, out_path=out)

    assert result == out
    assert out.is_file()
    assert "# Báo cáo phân tích — a.png" in out.read_text(encoding="utf-8")
    # artifact report.md cũng được ghi vào store
    assert (tmp_path / "art" / aid / "report.md").is_file()


def test_export_markdown_returns_store_path_without_out_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")
    svc = ReportService(store)
    aid = "abc123"
    store.put_bytes(aid, "report.json", json.dumps(_REP).encode("utf-8"))

    result = svc.export_markdown(aid)

    assert result == store.abs_path(f"{aid}/report.md")
    assert result.is_file()
