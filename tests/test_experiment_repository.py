"""Persistence tests for `ExperimentRepository`.

The repository currently exposes a single ``save`` (create) entry point, so
these tests exercise create + read-back across a fresh session (retrieve), the
empty-``per_method`` edge case, and ``default=str`` serialization of non-JSON
provenance values. There is no update/delete API on this repository today.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from imageguard.storage.db import make_session_factory
from imageguard.storage.models import Experiment, ExperimentResult
from imageguard.storage.repository import ExperimentRepository


def _count(session, model) -> int:
    return len(list(session.scalars(select(model))))


def test_save_creates_experiment_and_results(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "e.db")
    repo = ExperimentRepository(sf)

    repo.save({
        "experiment_id": "exp-abc",
        "dataset": "synthforge",
        "split": "dev",
        "provenance": {"seed": 42, "config_hash": "deadbeef"},
        "per_method": {"copy_move_keypoint": {"f1": 0.5, "n": 4}},
    })

    with sf() as s:  # session mới — đọc lại từ DB (retrieve)
        rows = list(s.scalars(select(Experiment)))
        assert len(rows) == 1
        exp = rows[0]
        assert exp.experiment_id == "exp-abc"
        assert exp.dataset == "synthforge"
        assert exp.split == "dev"
        assert json.loads(exp.provenance_json)["seed"] == 42

        results = list(s.scalars(select(ExperimentResult)))
        assert len(results) == 1
        assert results[0].method == "copy_move_keypoint"
        assert results[0].experiment_id == exp.id
        assert json.loads(results[0].metrics_json)["f1"] == 0.5


def test_save_with_empty_per_method(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "e.db")
    repo = ExperimentRepository(sf)

    repo.save({
        "experiment_id": "exp-empty",
        "dataset": "comofod",
        "split": "test",
        "provenance": {},
        "per_method": {},
    })

    with sf() as s:
        assert _count(s, Experiment) == 1
        assert _count(s, ExperimentResult) == 0


def test_save_stringifies_non_json_provenance(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "e.db")
    repo = ExperimentRepository(sf)

    repo.save({
        "experiment_id": "exp-path",
        "dataset": "synthforge",
        "split": "dev",
        "provenance": {"root": Path("reports")},
        "per_method": {},
    })

    with sf() as s:
        exp = s.scalar(select(Experiment))
        assert json.loads(exp.provenance_json)["root"] == "reports"


def test_multiple_saves_accumulate(tmp_path: Path) -> None:
    sf = make_session_factory(tmp_path / "e.db")
    repo = ExperimentRepository(sf)

    for i in range(3):
        repo.save({
            "experiment_id": f"exp-{i}",
            "dataset": "synthforge",
            "split": "dev",
            "provenance": {},
            "per_method": {"copy_move_keypoint": {"f1": float(i)}},
        })

    with sf() as s:
        assert _count(s, Experiment) == 3
        assert _count(s, ExperimentResult) == 3
