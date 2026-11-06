"""Validated folder-ingest configuration loaded with the existing PyYAML."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("data") / "ingest" / "config.yaml"
DEFAULT_METHODS = ("keypoint", "block", "ela")
_ALLOWED_METHODS = frozenset(
    {"keypoint", "block", "ela", "noise", "noise_local", "freq", "splicing"}
)
_ABSOLUTE_PATH_TEXT = re.compile(
    r"(?i)(?:[A-Z]:[\\/]|\\\\[^\\\s]+\\|(?:^|\s)/(?:[A-Za-z0-9._~-]+)(?:/|\\|$))"
)


class IngestConfigError(ValueError):
    """A missing or invalid folder-ingest configuration."""


@dataclass(frozen=True, slots=True)
class IngestConfig:
    camera_label: str = "unspecified"
    max_pending: int = 1
    methods: tuple[str, ...] = DEFAULT_METHODS
    data_dir: Path = Path("data")
    consent_note: str | None = None
    http: dict[str, Any] | None = None
    enhance_sibling: bool = False

    @property
    def custody_path(self) -> Path:
        return self.data_dir / "ingest" / "custody.jsonl"


def load_ingest_config(path: Path | str | None = None) -> IngestConfig:
    """Load an explicit config or the optional cwd-relative default file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is None and not config_path.is_file():
        return IngestConfig()
    if not config_path.is_file():
        raise IngestConfigError("config file does not exist")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise IngestConfigError("config file could not be read") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise IngestConfigError("config must be a YAML mapping")

    allowed = {
        "camera_label", "max_pending", "methods", "data_dir",
        "consent_note", "http", "enhance_sibling",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise IngestConfigError("config contains an unknown field")

    camera_label = raw.get("camera_label", "unspecified")
    if not isinstance(camera_label, str):
        raise IngestConfigError("camera_label must be a string")
    _validate_operator_text(camera_label, "camera_label")

    max_pending = raw.get("max_pending", 1)
    if type(max_pending) is not int or max_pending < 1:
        raise IngestConfigError("max_pending must be a positive integer")

    methods_value = raw.get("methods", list(DEFAULT_METHODS))
    if not isinstance(methods_value, list) or not methods_value:
        raise IngestConfigError("methods must be a non-empty list")
    if any(not isinstance(method, str) or not method for method in methods_value):
        raise IngestConfigError("methods must contain non-empty strings")
    if not set(methods_value) <= _ALLOWED_METHODS:
        raise IngestConfigError("methods contains an unsupported method")

    data_dir_value = raw.get("data_dir", "data")
    if not isinstance(data_dir_value, str) or not data_dir_value.strip():
        raise IngestConfigError("data_dir must be a relative path")
    data_dir = Path(data_dir_value)
    windows_data_dir = PureWindowsPath(data_dir_value)
    posix_data_dir = PurePosixPath(data_dir_value)
    if (
        data_dir.is_absolute()
        or windows_data_dir.is_absolute()
        or windows_data_dir.drive
        or ".." in windows_data_dir.parts
        or ".." in posix_data_dir.parts
    ):
        raise IngestConfigError("data_dir must stay relative to the working directory")

    consent_note = raw.get("consent_note")
    if consent_note is not None:
        if not isinstance(consent_note, str):
            raise IngestConfigError("consent_note must be a string")
        _validate_operator_text(consent_note, "consent_note")

    http = raw.get("http")
    if http is not None:
        if not isinstance(http, dict):
            raise IngestConfigError("http must be a mapping")
        if "enabled" in http and not isinstance(http["enabled"], bool):
            raise IngestConfigError("http.enabled must be a boolean")

    enhance_sibling = raw.get("enhance_sibling", False)
    if not isinstance(enhance_sibling, bool):
        raise IngestConfigError("enhance_sibling must be a boolean")

    return IngestConfig(
        camera_label=camera_label,
        max_pending=max_pending,
        methods=tuple(methods_value),
        data_dir=data_dir,
        consent_note=consent_note,
        http=http,
        enhance_sibling=enhance_sibling,
    )


def _validate_operator_text(value: str, field: str) -> None:
    if _ABSOLUTE_PATH_TEXT.search(value):
        raise IngestConfigError(f"{field} must not contain an absolute host path")
