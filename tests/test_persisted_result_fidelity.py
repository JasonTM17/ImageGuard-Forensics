"""Isolated SQLite and export fault points preserve saved forensic results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from imageguard.services.history_service import HistoryService
from imageguard.services.report_service import ReportService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository


@pytest.fixture
def history(tmp_path):
    sf = make_session_factory(tmp_path / "fixture.db")
    images, analyses = ImageRepository(sf), AnalysisRepository(sf)
    image = images.get_or_create("ab" * 32, "fixture.png", 80, 80, "png")
    report = {"tool_version": "0.1.0", "overall_verdict": "inconclusive", "limitations": [],
              "methods": [{"method": "copy_move_keypoint", "verdict": "inconclusive",
                           "score": 3.0, "score_definition": "verified pairs",
                           "evidence": {"n_verified": 3, "nested": {"area": 0.5}},
                           "limitations": ["weak evidence"], "duration_ms": 123}]}
    analyses.save_report("r1", image.id, report, "config", 150)
    yield HistoryService(analyses, ArtifactStore(tmp_path / "art")), sf
    sf.kw["bind"].dispose()


def test_history_get_restores_saved_method_detail(history):
    svc, _ = history
    method = svc.get("r1")["methods"][0]
    assert method["evidence"] == {"n_verified": 3, "nested": {"area": 0.5}}
    assert method["limitations"] == ["weak evidence"]
    assert method["duration_ms"] == 123
    assert method["score_definition"] == "verified pairs"


def test_real_analysis_persistence_round_trip_retains_method_fields(history, valid_png):
    from imageguard.services.analysis_service import (
        AnalysisRequest,
        AnalysisService,
        report_to_dict,
    )

    svc, sf = history
    analysis = AnalysisService(images=ImageRepository(sf), analyses=svc.analyses, store=svc.store)
    report = analysis.analyze(AnalysisRequest(valid_png, methods=["noise_local"], include_exif=False))
    saved = svc.get(report.report_id)["methods"][0]
    original = report_to_dict(report)["methods"][0]
    for key in ("method", "verdict", "score", "score_definition", "evidence", "limitations", "duration_ms"):
        assert saved[key] == original[key]


@pytest.mark.parametrize("evidence,limitations", [("null", "null"), ("[]", "{}"),
                                                 ("corrupt", "corrupt")])
def test_history_legacy_invalid_json_fields_have_renderable_defaults(history, evidence, limitations):
    from imageguard.storage.models import MethodResult

    svc, sf = history
    with sf() as session:
        method = session.query(MethodResult).one()
        method.evidence_json, method.limitations_json = evidence, limitations
        session.commit()
    method = svc.get("r1")["methods"][0]
    assert method["evidence"] == {}
    assert method["limitations"] == []
    assert method["duration_ms"] == 123


def test_reopened_history_dialog_displays_persisted_evidence(qtbot, history):
    from PySide6.QtWidgets import QLabel

    from imageguard.ui.screens.history import AnalysisDetailDialog

    svc, _ = history
    dialog = AnalysisDetailDialog(svc.get("r1"), svc)
    qtbot.addWidget(dialog)
    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert "123 ms" in labels
    assert "n_verified=3" in labels
    assert "weak evidence" in labels


@pytest.fixture
def report_service(tmp_path):
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("r1", "report.json", json.dumps({"report_id": "r1", "methods": []}).encode())
    store.put_bytes("r1", "mask.png", b"original-prediction-mask")
    store.put_bytes("r2", "report.json", b'{"report_id":"r2"}')
    return ReportService(store)


@pytest.mark.parametrize("destination", ["r1/report.json", "r1/mask.png", "r2/report.json",
                                         "r2/report.md"])
def test_report_export_rejects_artifact_collisions_before_writes(report_service, destination):
    svc = report_service
    root = svc.store.root
    path = svc.store.abs_path(destination)
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="artifact"):
        svc.export_markdown("r1", path)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("fault", ["flush", "replace"])
def test_external_export_failure_preserves_previous_file(report_service, tmp_path, monkeypatch, fault):
    svc = report_service
    target = tmp_path / "previous.md"
    target.write_bytes(b"previous-good-export")
    original_sync, original_replace = os.fsync, os.replace
    syncs = []

    def fail_sync(fd):
        syncs.append(fd)
        # The first sync belongs to the canonical artifact, the second to the external file.
        if len(syncs) == 2:
            raise OSError("controlled export flush failure")
        return original_sync(fd)

    def fail_replace(source, dest):
        if Path(dest) == target:
            raise OSError("controlled export replace failure")
        return original_replace(source, dest)

    monkeypatch.setattr(os, "fsync" if fault == "flush" else "replace",
                        fail_sync if fault == "flush" else fail_replace)
    with pytest.raises(OSError, match="controlled export"):
        svc.export_markdown("r1", target)
    assert target.read_bytes() == b"previous-good-export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["art", "previous.md"]
    assert svc.read_report_json("r1")["report_id"] == "r1"


def test_export_allows_unicode_external_and_canonical_markdown(report_service, tmp_path):
    svc = report_service
    external = tmp_path / "báo cáo mới.md"
    external.write_bytes(b"previous-export")
    assert svc.export_markdown("r1", external) == external
    canonical = svc.store.abs_path("r1/report.md")
    assert external.read_bytes() == canonical.read_bytes()
    assert svc.export_markdown("r1", canonical) == canonical
    assert canonical.read_text(encoding="utf-8").startswith("# Báo cáo")
    assert svc.read_report_json("r1")["report_id"] == "r1"


@pytest.mark.parametrize("entry", ["analysis", "history", "detail"])
def test_ui_export_entry_points_report_collision_without_changing_artifacts(qtbot, monkeypatch, history, entry):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from imageguard.services.analysis_service import AnalysisService
    from imageguard.ui.screens.analyze import AnalyzeScreen
    from imageguard.ui.screens.history import AnalysisDetailDialog, HistoryScreen

    svc, _ = history
    svc.store.put_bytes("r1", "report.json", b'{"report_id":"r1","methods":[]}')
    target = svc.store.abs_path("r1/report.json")
    before = target.read_bytes()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), "Markdown"))
    warnings, successes = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(_args[-1]))
    monkeypatch.setattr(QMessageBox, "information", lambda *_args: successes.append(_args[-1]))
    if entry == "analysis":
        screen = AnalyzeScreen(AnalysisService(store=svc.store))
        screen._report = SimpleNamespace(report_id="r1")
        qtbot.addWidget(screen)
        screen._export_report()
    elif entry == "history":
        screen = HistoryScreen(svc)
        qtbot.addWidget(screen)
        screen.table.selectRow(0)
        screen._export()
    else:
        screen = AnalysisDetailDialog(svc.get("r1"), svc)
        qtbot.addWidget(screen)
        screen._export()
    assert len(warnings) == 1
    assert "artifact" in warnings[0]
    assert successes == []
    assert target.read_bytes() == before
    assert not svc.store.abs_path("r1/report.md").exists()


@pytest.mark.parametrize("payload", [[], None, "not a report", 0])
def test_read_report_json_rejects_non_object_roots(report_service, payload):
    svc = report_service
    svc.store.put_bytes("r1", "report.json", json.dumps(payload).encode())
    assert svc.read_report_json("r1") is None
    assert svc.export_markdown("r1") is None
