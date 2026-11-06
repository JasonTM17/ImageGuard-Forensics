"""Append-only custody JSONL for folder ingest.

The log is not tamper-proof. Camera labels are operator-supplied, file
modification times are filesystem-unverified, and neither an authenticated
camera nor a verified capture time is established by these records.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class CustodyLog:
    """Append custody events without rewriting earlier entries."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def latest_events(self) -> dict[tuple[str, str], str]:
        """Read the last event for each basename and content hash."""
        latest: dict[tuple[str, str], str] = {}
        try:
            stream = self.path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return latest
        with stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                basename = record.get("basename")
                image_sha256 = record.get("image_sha256")
                event = record.get("event")
                if (
                    isinstance(basename, str)
                    and isinstance(image_sha256, str)
                    and isinstance(event, str)
                ):
                    latest[(basename, image_sha256)] = event
        return latest

    def append(
        self,
        *,
        event: str,
        image_sha256: str,
        basename: str,
        camera_label: str,
        file_mtime: str,
        analysis_id: str | None,
        config_hash: str,
        tool_version: str,
        consent_note: str | None = None,
        error_code: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "record_version": 1,
            "event_id": str(uuid4()),
            "event_time": datetime.now(UTC).isoformat(),
            "event": event,
            "image_sha256": image_sha256,
            "basename": basename,
            "camera_label": {
                "value": camera_label,
                "trust": "operator_supplied",
            },
            "file_mtime": {
                "value": file_mtime,
                "trust": "filesystem_unverified",
            },
            "analysis_id": analysis_id,
            "config_hash": config_hash,
            "tool_version": tool_version,
        }
        if consent_note is not None:
            record["consent_note"] = consent_note
        if error_code is not None:
            record["error_code"] = error_code

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
