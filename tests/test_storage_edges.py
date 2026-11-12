"""Storage edge cases — artifact-store guards and SQLite persistence.

Covers only behaviour stated in `docs/database.md`,
`docs/security-error-handling.md` and the storage module contracts:

- `ArtifactStore` naming allowlist (`[a-z0-9_-]`, filename `+ .ext`), path
  containment after `resolve()`, overwrite semantics, delete scoping.
- `make_session_factory` schema creation idempotence and the documented
  `SCHEMA_VERSION` mismatch error.
- `AnalysisRepository`/`ImageRepository` round trips (null/empty evidence,
  very long text, unicode, injection-looking literals), history ordering with
  `limit`, cascade delete, and thread-level session isolation.

The repository persists report dicts; there is no DB -> `ForensicReport`
reconstruction API, so "round trip" here means save through the repository and
read the persisted columns back in a fresh session.

All files live under `tmp_path`; no network, no real data directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.models import (
    SCHEMA_VERSION,
    Analysis,
    Artifact,
    ImageAsset,
    MetaKV,
    MethodResult,
)
from imageguard.storage.repository import (
    AnalysisRepository,
    ImageRepository,
    _assert_relative,
)

_SHA = "ab" * 32


# --------------------------------------------------------------------------
# helpers / fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "edge.db"


@pytest.fixture()
def sf(db_path: Path) -> sessionmaker[Session]:
    return make_session_factory(db_path)


def _method(
    method: str = "ela",
    *,
    evidence: object,
    verdict: str = "no_indication",
    score: float | None = 0.0,
    score_definition: str = "p99 heatmap",
    limitations: list[str] | None = None,
    duration_ms: int = 3,
) -> dict:
    return {
        "method": method,
        "verdict": verdict,
        "score": score,
        "score_definition": score_definition,
        "evidence": evidence,
        "limitations": limitations if limitations is not None else [],
        "duration_ms": duration_ms,
    }


def _report(
    *methods: dict,
    verdict: str = "inconclusive",
    limitations: list[str] | None = None,
) -> dict:
    return {
        "tool_version": "0.1.0",
        "overall_verdict": verdict,
        "limitations": limitations if limitations is not None else [],
        "methods": list(methods),
    }


def _seed_image(images: ImageRepository) -> ImageAsset:
    return images.get_or_create(_SHA, "seed.png", 8, 8, "png")


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def _resolved(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


# --------------------------------------------------------------------------
# ArtifactStore — name sanitizer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "analysis_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "..\\escape",
        "/absolute",
        "\\absolute",
        "C:\\absolute",
        "C:/absolute",
        "C:relative",
        "a/b",
        "a\\b",
        "a b",
        "a.b",
        "a:b",
        "ảnh",
    ],
)
def test_analysis_id_rejects_traversal_absolute_and_non_allowlisted(
    analysis_id: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.rel_path(analysis_id, "report.json")


@pytest.mark.parametrize(
    "filename",
    [
        "",
        ".",
        "..",
        "../escape.png",
        "..\\escape.png",
        "/escape.png",
        "\\escape.png",
        "C:\\escape.png",
        "C:/escape.png",
        "sub/escape.png",
        "sub\\escape.png",
        "a.b.c",
        ".hidden.png",
        "report.",
        "report ",
        "report.abcdefghi",
        "ảnh.png",
        "tab\tname.png",
    ],
)
def test_filename_rejects_traversal_absolute_and_bad_extension(
    filename: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.rel_path("aid", filename)


def test_sanitizer_accepts_documented_allowlist_shapes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")

    assert store.rel_path("aid_1-x", "report.json") == "aid_1-x/report.json"
    assert store.rel_path("AID", "MASK.PNG") == "AID/MASK.PNG"
    # filename may omit the extension (a dot is optional).
    assert store.rel_path("aid", "mask") == "aid/mask"


def test_reserved_device_names_pass_shape_guard_but_dot_space_forms_do_not(
    tmp_path: Path,
) -> None:
    """Allowlist is shape-based, so bare device names are not special-cased.

    Only the classic trailing-dot / trailing-space device-name variants are
    rejected.  This test records the documented allowlist behaviour; it does
    not claim device names are safe to write on every Windows build.
    """
    store = ArtifactStore(tmp_path / "art")

    assert store.rel_path("CON", "nul.png") == "CON/nul.png"
    assert store.rel_path("aid", "AUX") == "aid/AUX"
    assert store.rel_path("aid", "COM1.png") == "aid/COM1.png"

    for name in ("CON.", "CON ", "NUL.", "LPT1 "):
        with pytest.raises(ValueError):
            store.rel_path("aid", name)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "sanitizer regex uses $, which also matches before a trailing newline, "
        "so 'aid\\n' and 'report.json\\n' pass the documented [a-z0-9_-] "
        "allowlist instead of raising ValueError; put_bytes then fails with "
        "OSError from the filesystem"
    ),
)
@pytest.mark.parametrize(
    ("analysis_id", "filename"),
    [("aid\n", "report.json"), ("aid", "report.json\n")],
    ids=["analysis_id", "filename"],
)
def test_sanitizer_rejects_trailing_newline(
    analysis_id: str, filename: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.rel_path(analysis_id, filename)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "re.IGNORECASE folds non-ASCII lookalikes into [a-z], so KELVIN SIGN "
        "and LONG S are accepted although the documented allowlist is ASCII"
    ),
)
@pytest.mark.parametrize(
    ("analysis_id", "filename"),
    [("\u212a", "report.json"), ("\u017fon", "report.json"),
     ("aid", "x.\u212a")],
    ids=["kelvin-id", "long-s-id", "kelvin-ext"],
)
def test_sanitizer_rejects_non_ascii_case_fold_lookalikes(
    analysis_id: str, filename: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.rel_path(analysis_id, filename)


# --------------------------------------------------------------------------
# ArtifactStore — containment / overwrite / delete
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "C:x",
        "sub/../report.json",
        "./report.json",
        "con/report.json",
        "aid/../other/report.json",
    ],
)
def test_abs_path_only_ever_returns_paths_inside_root(
    rel: str, tmp_path: Path
) -> None:
    """Suspicious-but-resolvable rels must stay under the resolved root."""
    root = tmp_path / "art"
    store = ArtifactStore(root)

    try:
        path = store.abs_path(rel)
    except ValueError:
        return  # rejecting is always acceptable

    assert _resolved(path).startswith(_resolved(root) + os.sep)


def test_abs_path_normalises_parent_segments_that_stay_inside(
    tmp_path: Path,
) -> None:
    root = tmp_path / "art"
    store = ArtifactStore(root)

    path = store.abs_path("sub/../report.json")

    assert path == (root / "report.json").resolve()


@pytest.mark.parametrize(
    "rel",
    [
        "",
        "..",
        "../escape",
        "a/../../escape",
        "/absolute",
        "C:/absolute",
        "\\\\server\\share\\file.png",
    ],
)
def test_abs_path_rejects_empty_absolute_and_escaping_rels(
    rel: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.abs_path(rel)


def test_put_bytes_overwrite_keeps_latest_and_no_temp_files(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "art")

    first = store.put_bytes("aid", "report.json", b"first")
    second = store.put_bytes("aid", "report.json", b"second")

    assert first == second == "aid/report.json"
    assert store.read(first) == b"second"
    names = {p.name for p in (tmp_path / "art" / "aid").iterdir()}
    assert names == {"report.json"}


def test_put_bytes_round_trip_unicode_and_empty_payload(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")
    payload = "báo cáo — 日本語 🎭".encode()
    store.put_bytes("aid", "report.txt", payload)
    store.put_bytes("aid", "empty.bin", b"")

    assert store.read("aid/report.txt") == payload
    assert store.read("aid/empty.bin") == b""


def test_put_bytes_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("aid", "report.json", b"original")

    def _boom(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("imageguard.storage.artifacts.os.replace", _boom)
    with pytest.raises(OSError):
        store.put_bytes("aid", "report.json", b"replacement")

    assert store.read("aid/report.json") == b"original"
    assert {p.name for p in (tmp_path / "art" / "aid").iterdir()} == {
        "report.json"
    }


def test_delete_analysis_removes_only_target_tree(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "art")
    store.put_bytes("aid1", "report.json", b"{}")
    store.put_bytes("aid2", "report.json", b"{}")
    sibling = tmp_path / "art-evil"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")

    store.delete_analysis("aid1")

    assert not (tmp_path / "art" / "aid1").exists()
    assert (tmp_path / "art" / "aid2" / "report.json").exists()
    # A sibling directory sharing the root's name prefix must stay untouched.
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"

    store.delete_analysis("missing")  # unknown id is a no-op
    store.delete_analysis("art-evil")  # valid shape, but not under root
    assert (sibling / "keep.txt").exists()


@pytest.mark.parametrize(
    "analysis_id",
    ["", ".", "..", "../escape", "/absolute", "C:\\absolute", "a/b", "ảnh"],
)
def test_delete_analysis_rejects_unsafe_ids(
    analysis_id: str, tmp_path: Path
) -> None:
    store = ArtifactStore(tmp_path / "art")
    with pytest.raises(ValueError):
        store.delete_analysis(analysis_id)


# --------------------------------------------------------------------------
# Schema creation
# --------------------------------------------------------------------------


def test_schema_creation_is_idempotent(
    db_path: Path, sf: sessionmaker[Session]
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    analyses.save_report(
        "00000000-0000-4000-8000-000000000001", asset.id,
        _report(_method(evidence={})), "hash", 5,
    )

    # Second factory over an existing DB must not drop or duplicate anything.
    sf2 = make_session_factory(db_path)
    analyses2 = AnalysisRepository(sf2)

    with sf2() as s:
        assert _count(s, MetaKV) == 1
        version = s.get(MetaKV, "schema_version")
        assert version is not None and version.value == str(SCHEMA_VERSION)
        assert _count(s, Analysis) == 1

    got = analyses2.get("00000000-0000-4000-8000-000000000001")
    assert got is not None and got.image is not None
    assert got.image.file_name == "seed.png"


def test_schema_version_mismatch_raises_without_silent_recreate(
    db_path: Path, sf: sessionmaker[Session]
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    analyses.save_report(
        "00000000-0000-4000-8000-000000000002", asset.id,
        _report(_method(evidence={})), "hash", 5,
    )

    con = sqlite3.connect(db_path)
    try:
        con.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        con.commit()
    finally:
        con.close()

    with pytest.raises(RuntimeError, match="schema_version"):
        make_session_factory(db_path)

    # Documented behaviour: fail loudly, never migrate/recreate silently.
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT value FROM meta").fetchall()
        n_analyses = con.execute(
            "SELECT COUNT(*) FROM analyses").fetchone()[0]
    finally:
        con.close()
    assert rows == [("999",)]
    assert n_analyses == 1


def test_foreign_keys_enforced_on_pooled_connections(
    sf: sessionmaker[Session],
) -> None:
    for _ in range(2):  # different pooled connections
        with sf() as s:
            assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1

    with pytest.raises(IntegrityError), sf() as s:
        s.add(MethodResult(
            analysis_id="does-not-exist", method="ela",
            verdict="no_indication", score=None, score_definition="",
            evidence_json="{}", limitations_json="[]", duration_ms=0,
        ))
        s.commit()


# --------------------------------------------------------------------------
# Repository — report round trip, long/unicode/injection text
# --------------------------------------------------------------------------


def test_report_round_trip_with_none_and_empty_evidence(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = images.get_or_create(_SHA, "ảnh_日本語_🎭.png", 32, 16, "png")

    limitations = ["hạn chế phương pháp", ""]
    report = _report(
        _method("ela", evidence=None, verdict="inconclusive"),
        _method("exif_audit", evidence={}, score=None, score_definition=""),
        _method("noise_inconsistency", evidence={
            "ghi_chú": "miền tần số", "nested": [1, None, "x"],
        }),
        limitations=limitations,
    )
    analyses.save_report(
        "00000000-0000-4000-8000-000000000010", asset.id, report,
        "deadbeefdeadbeef", 42,
    )

    got = analyses.get("00000000-0000-4000-8000-000000000010")

    assert got is not None
    assert got.image is not None and got.image.file_name == "ảnh_日本語_🎭.png"
    assert got.tool_version == "0.1.0"
    assert got.overall_verdict == "inconclusive"
    assert got.config_hash == "deadbeefdeadbeef"
    assert got.duration_ms == 42
    assert json.loads(got.limitations_json) == limitations
    assert got.artifacts == []

    by_method = {m.method: m for m in got.method_results}
    assert set(by_method) == {"ela", "exif_audit", "noise_inconsistency"}
    assert json.loads(by_method["ela"].evidence_json) is None
    assert json.loads(by_method["exif_audit"].evidence_json) == {}
    assert json.loads(by_method["noise_inconsistency"].evidence_json) == {
        "ghi_chú": "miền tần số", "nested": [1, None, "x"],
    }
    assert by_method["exif_audit"].score is None
    assert by_method["ela"].score == 0.0


def test_save_report_without_methods(sf: sessionmaker[Session]) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    aid = "00000000-0000-4000-8000-000000000011"

    analyses.save_report(aid, asset.id, _report(), "hash", 0)

    got = analyses.get(aid)
    assert got is not None
    assert got.method_results == []
    assert got.limitations_json == "[]"


def test_text_fields_round_trip_very_long_strings(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    long_ascii = "x" * 20_000
    long_unicode = "ả" * 5_000
    aid = "00000000-0000-4000-8000-000000000012"

    report = _report(
        _method(
            evidence={"blob": long_ascii},
            score_definition="d" * 5_000,
            limitations=[long_unicode],
        ),
        limitations=[long_ascii, long_unicode],
    )
    analyses.save_report(aid, asset.id, report, "hash", 1)

    got = analyses.get(aid)
    assert got is not None
    limitations = json.loads(got.limitations_json)
    assert limitations == [long_ascii, long_unicode]
    assert len(limitations[0]) == 20_000
    method = got.method_results[0]
    assert json.loads(method.evidence_json)["blob"] == long_ascii
    assert len(method.score_definition) == 5_000


def test_injection_looking_strings_are_stored_verbatim(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    inject = "'; DROP TABLE analyses; --"
    asset = images.get_or_create(
        "cd" * 32, f"{inject}.png", 8, 8, "png")
    aid = "'; DELETE FROM method_results; --"[:36]

    analyses.save_report(
        aid, asset.id,
        _report(_method(evidence={"query": inject, "list": [inject]}),
                verdict=inject, limitations=[inject]),
        "hash", 2,
    )

    got = analyses.get(aid)
    assert got is not None
    assert got.overall_verdict == inject
    assert got.image is not None
    assert got.image.file_name == f"{inject}.png"
    assert json.loads(got.limitations_json) == [inject]
    assert json.loads(got.method_results[0].evidence_json) == {
        "query": inject, "list": [inject],
    }

    # Nothing was executed: every table still exists and row counts are sane.
    with sf() as s:
        tables = set(s.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'")).scalars())
        assert {"analyses", "method_results", "artifacts",
                "image_assets", "meta"} <= tables
        assert _count(s, Analysis) == 1
        assert _count(s, ImageAsset) == 1
    assert len(analyses.list()) == 1


# --------------------------------------------------------------------------
# Repository — history ordering, detached rows, artifacts, delete
# --------------------------------------------------------------------------


def test_history_ordering_and_limit_pagination(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    ids = [f"00000000-0000-4000-8000-{i:012d}" for i in range(5)]

    for aid in ids:
        analyses.save_report(aid, asset.id, _report(_method(evidence={})),
                             "hash", 1)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    with sf() as s:
        for offset, aid in enumerate(ids):
            row = s.get(Analysis, aid)
            assert row is not None
            row.created_at = start + timedelta(minutes=offset)
        s.commit()

    newest_first = list(reversed(ids))
    assert [a.id for a in analyses.list()] == newest_first
    assert [a.id for a in analyses.list(limit=2)] == newest_first[:2]
    assert analyses.list(limit=0) == []
    assert len(analyses.list(limit=50)) == len(ids)


def test_list_returns_detached_rows_with_eager_relations(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    aid = "00000000-0000-4000-8000-000000000020"
    analyses.save_report(
        aid, asset.id,
        _report(_method("copy_move_keypoint", evidence={}),
                _method("ela", evidence={})),
        "hash", 1,
    )

    rows = analyses.list()

    # Session is closed here; the documented eager loads must still work.
    assert len(rows) == 1
    assert rows[0].image.file_name == "seed.png"
    assert len(rows[0].method_results) == 2
    assert {m.method for m in rows[0].method_results} == {
        "copy_move_keypoint", "ela",
    }


@pytest.mark.parametrize(
    "rel_path",
    [
        "../escape.png",
        "a/../../escape.png",
        "C:/absolute.png",
        "\\\\server\\share\\x.png",
    ],
)
def test_add_artifact_rejects_unsafe_rel_paths(
    rel_path: str, sf: sessionmaker[Session]
) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    aid = "00000000-0000-4000-8000-000000000021"
    analyses.save_report(aid, asset.id, _report(_method(evidence={})),
                         "hash", 1)

    with pytest.raises(ValueError):
        analyses.add_artifact(aid, "report", "", rel_path)

    with sf() as s:
        assert _count(s, Artifact) == 0

    analyses.add_artifact(aid, "report", "", f"{aid}/report.json")
    got = analyses.get(aid)
    assert got is not None
    assert [a.rel_path for a in got.artifacts] == [f"{aid}/report.json"]


def test_assert_relative_accepts_safe_paths() -> None:
    _assert_relative("aid/report.json")
    _assert_relative("aid/mask.png")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "on Windows Path('/x').is_absolute() is False for drive-less rooted "
        "paths, so _assert_relative stores them even though docs/database.md "
        "promises absolute paths are blocked; ArtifactStore.abs_path cannot "
        "resolve such a row"
    ),
)
@pytest.mark.parametrize(
    "rel_path", ["/absolute.png", "\\absolute.png"],
    ids=["forward-slash", "backslash"],
)
def test_assert_relative_rejects_drive_less_rooted_paths(
    rel_path: str,
) -> None:
    with pytest.raises(ValueError):
        _assert_relative(rel_path)


def test_delete_cascades_dependent_rows(sf: sessionmaker[Session]) -> None:
    images = ImageRepository(sf)
    analyses = AnalysisRepository(sf)
    asset = _seed_image(images)
    aid = "00000000-0000-4000-8000-000000000022"
    analyses.save_report(aid, asset.id, _report(_method(evidence={})),
                         "hash", 1)
    analyses.add_artifact(aid, "mask", "ela", f"{aid}/ela_mask.png")
    analyses.add_artifact(aid, "report", "", f"{aid}/report.json")

    rels = analyses.delete(aid)

    assert set(rels) == {f"{aid}/ela_mask.png", f"{aid}/report.json"}
    assert analyses.get(aid) is None
    with sf() as s:
        assert _count(s, Analysis) == 0
        assert _count(s, MethodResult) == 0
        assert _count(s, Artifact) == 0
        # Image rows are deduplicated by sha256 and survive analysis deletes.
        assert _count(s, ImageAsset) == 1
    assert analyses.delete(aid) == []


# --------------------------------------------------------------------------
# Repository — concurrent sessions in one process
# --------------------------------------------------------------------------


def test_concurrent_sessions_insert_distinct_rows_without_corruption(
    sf: sessionmaker[Session],
) -> None:
    images = ImageRepository(sf)
    asset = images.get_or_create("ef" * 32, "shared.png", 8, 8, "png")
    n_threads, per_thread = 4, 6
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(t: int) -> None:
        try:
            barrier.wait(timeout=10)
            analyses = AnalysisRepository(sf)
            for i in range(per_thread):
                aid = f"00000000-0000-4000-8000-{t:02d}{i:010d}"
                analyses.save_report(
                    aid, asset.id,
                    _report(_method(evidence={"worker": t})), "hash", i,
                )
                analyses.add_artifact(
                    aid, "report", "", f"{aid}/report.json")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not any(t.is_alive() for t in threads)
    assert errors == []

    with sf() as s:
        assert _count(s, Analysis) == n_threads * per_thread
        assert _count(s, MethodResult) == n_threads * per_thread
        assert _count(s, Artifact) == n_threads * per_thread
        ids = list(s.scalars(select(Analysis.id)))
    assert len(set(ids)) == n_threads * per_thread

    rows = AnalysisRepository(sf).list(limit=100)
    assert len(rows) == n_threads * per_thread
    for row in rows:
        assert len(row.method_results) == 1
