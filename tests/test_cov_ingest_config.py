"""Coverage gaps: ingest.config.load_ingest_config validation branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from imageguard.ingest.config import IngestConfig, IngestConfigError, load_ingest_config


def test_load_ingest_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestConfigError, match="does not exist"):
        load_ingest_config(tmp_path / "missing.yaml")


def test_load_ingest_config_defaults_when_default_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_ingest_config()
    assert isinstance(cfg, IngestConfig)
    assert cfg.methods == ("keypoint", "block", "ela")
    assert cfg.data_dir == Path("data")


def test_load_ingest_config_happy_path_and_custody_path(tmp_path: Path) -> None:
    yaml_text = """
camera_label: lab_cam_1
max_pending: 2
methods: [keypoint, ela]
data_dir: ingest_data
consent_note: operator approved
enhance_sibling: true
http:
  enabled: false
"""
    path = tmp_path / "ingest.yaml"
    path.write_text(yaml_text.strip() + "\n", encoding="utf-8")
    cfg = load_ingest_config(path)

    assert cfg.camera_label == "lab_cam_1"
    assert cfg.max_pending == 2
    assert cfg.methods == ("keypoint", "ela")
    assert cfg.data_dir == Path("ingest_data")
    assert cfg.consent_note == "operator approved"
    assert cfg.enhance_sibling is True
    assert cfg.http == {"enabled": False}
    assert cfg.custody_path == Path("ingest_data") / "ingest" / "custody.jsonl"


@pytest.mark.parametrize(
    ("body", "pattern"),
    [
        ("unknown_key: 1\n", "unknown field"),
        ("camera_label: 1\n", "camera_label must be a string"),
        ("max_pending: 0\n", "max_pending must be a positive integer"),
        ("methods: []\n", "methods must be a non-empty list"),
        ("methods: [bogus]\n", "unsupported method"),
        ('data_dir: "C:/abs"\n', "data_dir must stay relative"),
        ("consent_note: 9\n", "consent_note must be a string"),
        ("http: 1\n", "http must be a mapping"),
        ("enhance_sibling: 1\n", "enhance_sibling must be a boolean"),
        ('camera_label: "see D:\\\\secrets"\n', "must not contain an absolute host path"),
    ],
)
def test_load_ingest_config_validation_errors(
    tmp_path: Path, body: str, pattern: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(IngestConfigError, match=pattern):
        load_ingest_config(path)


def test_load_ingest_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(IngestConfigError, match="must be a YAML mapping"):
        load_ingest_config(path)
