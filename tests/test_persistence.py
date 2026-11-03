"""Storage layer — schema, dedup, relative-path invariant, cascade delete."""

from __future__ import annotations

from pathlib import Path

import pytest

from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import (
    AnalysisRepository,
    ImageRepository,
    _assert_relative,
)


@pytest.fixture()
def repos(tmp_path: Path):
    sf = make_session_factory(tmp_path / "t.db")
    return (ImageRepository(sf), AnalysisRepository(sf),
            ArtifactStore(tmp_path / "art"))


def test_dedup_image_by_sha256(repos) -> None:
    images, _, _ = repos
    a = images.get_or_create("ab" * 32, "x.png", 10, 10, "png")
    b = images.get_or_create("ab" * 32, "y.png", 10, 10, "png")
    assert a.id == b.id  # cùng sha256 → cùng row, file_name giữ lần đầu


def test_relative_path_guard() -> None:
    with pytest.raises(ValueError):
        _assert_relative("C:/abs/path.png")
    with pytest.raises(ValueError):
        _assert_relative("../escape.png")
    _assert_relative("abc/mask.png")  # ok


def test_artifact_store_traversal_guard(repos) -> None:
    _, _, store = repos
    with pytest.raises(ValueError):
        store.put_bytes("ok", "..\\evil.png", b"x")
    rel = store.put_bytes("abc123", "report.json", b"{}")
    assert rel == "abc123/report.json"
    with pytest.raises(ValueError):
        store.abs_path("../outside")


def test_delete_cascades(repos, tmp_path: Path) -> None:
    images, analyses, store = repos
    asset = images.get_or_create("cd" * 32, "i.png", 8, 8, "png")
    rd = {"tool_version": "0.1.0", "overall_verdict": "no_indication",
          "limitations": [], "methods": [{
              "method": "copy_move_keypoint", "verdict": "no_indication",
              "score": 0.0, "score_definition": "d", "evidence": {},
              "limitations": [], "duration_ms": 5}]}
    analyses.save_report("aid-1", asset.id, rd, "h", 5)
    analyses.add_artifact("aid-1", "report", "", "aid-1/report.json")
    store.put_bytes("aid-1", "report.json", b"{}")
    rels = analyses.delete("aid-1")
    assert "aid-1/report.json" in rels
    assert analyses.get("aid-1") is None
    store.delete_analysis("aid-1")
    assert not (tmp_path / "art" / "aid-1").exists()
