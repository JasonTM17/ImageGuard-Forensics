"""Regression: partial-failure ordering across storage/services.

7 fix: history delete dọn file TRƯỚC row; get_or_create nhận thua
IntegrityError; export guard chặn cả data_root ngoài artifacts; evaluation
ghi DB trước file; created_at default naive UTC; add_artifact từ chối
rel_path rỗng. SQLite + ArtifactStore thật trong tmp_path.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import SAWarning

import imageguard.services.evaluation_service as evs
from imageguard.services.evaluation_service import EvaluationService
from imageguard.services.history_service import HistoryService
from imageguard.services.report_service import ReportService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.models import (
    Analysis,
    Artifact,
    Experiment,
    ImageAsset,
)
from imageguard.storage.repository import AnalysisRepository, ImageRepository


@pytest.fixture()
def env(tmp_path: Path, ts):
    sf = make_session_factory(tmp_path / "app.db")
    store = ArtifactStore(tmp_path / "art")
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    yield SimpleNamespace(
        tmp=tmp_path, sf=sf, store=store, images=images, analyses=analyses)
    sf.kw["bind"].dispose()  # Windows: nhả file SQLite trước khi xoá tmp


def _seed_analysis(env, rid: str = "r1") -> None:
    asset = env.images.get_or_create("ab" * 32, "a.png", 8, 8, "PNG")
    env.analyses.save_report(rid, asset.id, {
        "tool_version": "t",
        "overall_verdict": "inconclusive",
        "limitations": [],
        "methods": [],
    }, "h", 5)


def test_locked_artifact_file_keeps_history_row(
        env, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_analysis(env)
    env.store.put_bytes("r1", "note.txt", b"x")
    svc = HistoryService(env.analyses, env.store)

    def locked(rid: str) -> None:
        raise PermissionError("controlled file lock")

    monkeypatch.setattr(env.store, "delete_analysis", locked)
    with pytest.raises(PermissionError):
        svc.delete("r1")
    # File khoá → raise TRƯỚC khi xoá row: row còn nguyên để retry.
    assert env.analyses.get("r1") is not None


def test_delete_cleans_files_then_row(env) -> None:
    _seed_analysis(env)
    env.store.put_bytes("r1", "note.txt", b"x")
    svc = HistoryService(env.analyses, env.store)

    assert svc.delete("r1") is True
    assert env.analyses.get("r1") is None
    assert not (env.store.root / "r1").exists()


def test_get_or_create_recovers_from_sha256_race(env) -> None:
    sha = "cd" * 32
    fired = False

    def _race(mapper, connection, target) -> None:
        # Giả lập bên thắng cuộc đua chèn cùng sha256 trên CÙNG connection:
        # ORM insert của get_or_create sẽ đụng UNIQUE constraint. Winner phải
        # commit độc lập — nếu không, rollback của bên thua xoá luôn row này.
        nonlocal fired
        if fired:
            return
        fired = True
        connection.execute(text(
            "INSERT INTO image_assets (sha256, file_name, width, height, "
            "format, created_at) VALUES (:s, 'race.png', 1, 1, 'PNG', "
            "'2026-01-01 00:00:00')"), {"s": target.sha256})
        connection.commit()

    event.listen(ImageAsset, "before_insert", _race)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SAWarning)
            row = env.images.get_or_create(sha, "race.png", 1, 1, "PNG")
    finally:
        event.remove(ImageAsset, "before_insert", _race)
    assert row.sha256 == sha
    assert row.id is not None
    assert row.file_name == "race.png"


def test_export_guard_blocks_data_root_outside_artifacts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # source mode: data_root() = ./data
    store = ArtifactStore(tmp_path / "data" / "artifacts")
    canonical = (store.root / "r1" / "report.md").resolve()
    victim = tmp_path / "data" / "backup.db"
    victim.parent.mkdir(parents=True, exist_ok=True)
    svc = ReportService(store)

    with pytest.raises(ValueError):
        svc._guard_export_destination(victim, canonical)
    svc._guard_export_destination(canonical, canonical)  # chính nó: ok
    svc._guard_export_destination(tmp_path / "out" / "x.md", canonical)  # ngoài data: ok


def test_add_artifact_rejects_empty_rel_path(env) -> None:
    _seed_analysis(env)
    with pytest.raises(ValueError):
        env.analyses.add_artifact("r1", "mask", "", "")
    with env.sf() as s:
        assert s.scalar(select(func.count(Artifact.id))) == 0


def test_evaluation_service_saves_db_before_files(
        tmp_path: Path, ts, monkeypatch: pytest.MonkeyPatch) -> None:
    flags = {"files_written": False}

    def _save(out_dir) -> None:
        flags["files_written"] = True
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval-e1.json").write_text("{}", encoding="utf-8")

    stub = SimpleNamespace(
        experiment_id="e1", dataset="synth", split="dev",
        provenance={"seed": 42}, per_method={"keypoint": {"f1": 0.5}},
        save=_save)

    class _FailingExperiments:
        def save(self, record: dict) -> None:
            raise OSError("controlled db fault")

    monkeypatch.setattr(evs, "SynthAdapter", lambda root: object())
    monkeypatch.setattr(evs, "load_method", lambda spec: object())
    monkeypatch.setattr(evs, "run_eval", lambda *a, **k: stub)
    svc = EvaluationService(ts, _FailingExperiments(),
                            reports_dir=tmp_path / "reports")

    with pytest.raises(OSError):
        svc.run("synth", tmp_path, "dev", ["keypoint"])
    # DB ghi trước → DB fail thì file chưa được viết (không để eval-* mồ côi).
    assert flags["files_written"] is False


def test_created_at_defaults_are_naive_utc() -> None:
    for model in (ImageAsset, Analysis, Experiment):
        value = model.__table__.c.created_at.default.arg(None)
        assert value.tzinfo is None
