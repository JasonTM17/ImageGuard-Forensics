"""Unit tests for `EvaluationService.run` routing and experiment persistence.

`run` selects ``SynthAdapter`` vs ``CoMoFoDAdapter`` by dataset name, builds
methods via ``load_method``, delegates to ``run_eval``, saves the report, and —
when an experiment repository is wired in — persists a summary row.
"""

from __future__ import annotations

from pathlib import Path

from imageguard.services import evaluation_service


class _FakeReport:
    def __init__(self) -> None:
        self.experiment_id = "exp-1"
        self.dataset = "synthforge"
        self.split = "dev"
        self.provenance = {"seed": 42}
        self.per_method = {"copy_move_keypoint": {"f1": 0.5}}
        self.saved_to: Path | None = None

    def save(self, out_dir: Path) -> None:
        self.saved_to = out_dir


class _FakeExperiments:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save(self, record: dict) -> None:
        self.saved.append(record)


def _patch(monkeypatch) -> dict:
    calls: dict = {}

    def fake_run_eval(adapter, methods, cfg, split, seed):
        calls["adapter"] = adapter
        calls["methods"] = methods
        calls["cfg"] = cfg
        calls["split"] = split
        calls["seed"] = seed
        return _FakeReport()

    monkeypatch.setattr(
        evaluation_service, "SynthAdapter", lambda root: ("SYNTH", root)
    )
    monkeypatch.setattr(
        evaluation_service, "CoMoFoDAdapter", lambda root: ("COMOFOD", root)
    )
    monkeypatch.setattr(evaluation_service, "load_method", lambda s: f"M:{s}")
    monkeypatch.setattr(evaluation_service, "run_eval", fake_run_eval)
    return calls


def test_run_routes_synth_dataset(tmp_path: Path, ts, monkeypatch) -> None:
    calls = _patch(monkeypatch)
    svc = evaluation_service.EvaluationService(ts, None, reports_dir=tmp_path)

    report = svc.run("synth", tmp_path, "dev", ["keypoint"], seed=7)

    assert calls["adapter"] == ("SYNTH", tmp_path)
    assert calls["methods"] == ["M:keypoint"]
    assert calls["split"] == "dev"
    assert calls["seed"] == 7
    assert isinstance(report, _FakeReport)
    assert report.saved_to == tmp_path  # report.save được gọi


def test_run_routes_comofod_dataset(tmp_path: Path, ts, monkeypatch) -> None:
    calls = _patch(monkeypatch)
    svc = evaluation_service.EvaluationService(ts, None, reports_dir=tmp_path)

    svc.run("comofod", tmp_path, "test", ["block", "ela"], seed=3)

    assert calls["adapter"] == ("COMOFOD", tmp_path)
    assert calls["methods"] == ["M:block", "M:ela"]
    assert calls["split"] == "test"


def test_run_saves_experiment_when_repository_wired(
    tmp_path: Path, ts, monkeypatch
) -> None:
    _patch(monkeypatch)
    experiments = _FakeExperiments()
    svc = evaluation_service.EvaluationService(ts, experiments, reports_dir=tmp_path)

    report = svc.run("synth", tmp_path, "dev", ["keypoint"], seed=7)

    assert len(experiments.saved) == 1
    saved = experiments.saved[0]
    assert saved["experiment_id"] == report.experiment_id
    assert saved["dataset"] == "synthforge"
    assert saved["split"] == "dev"
    assert saved["provenance"] == {"seed": 42}
    assert saved["per_method"] == {"copy_move_keypoint": {"f1": 0.5}}


def test_run_skips_save_when_no_repository(tmp_path: Path, ts, monkeypatch) -> None:
    _patch(monkeypatch)
    svc = evaluation_service.EvaluationService(ts, None, reports_dir=tmp_path)

    report = svc.run("synth", tmp_path, "dev", ["keypoint"], seed=7)

    assert isinstance(report, _FakeReport)
    assert report.saved_to == tmp_path  # vẫn save report, chỉ bỏ qua experiment
