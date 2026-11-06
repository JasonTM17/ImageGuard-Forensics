"""Headless CLI entry points for folder ingest."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from imageguard.core.config import load_thresholds
from imageguard.ingest.config import IngestConfigError, load_ingest_config
from imageguard.ingest.folder import FolderIngestor
from imageguard.services.analysis_service import AnalysisService
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.db import make_session_factory
from imageguard.storage.repository import AnalysisRepository, ImageRepository


def run_ingest(
    mode: str,
    directory: Path | str,
    config_path: Path | str | None = None,
) -> int:
    """Run one folder scan or poll until interrupted; return a process code."""
    try:
        ingest_config = load_ingest_config(config_path)
    except IngestConfigError:
        print("error: invalid ingest configuration", file=sys.stderr)
        return 2

    watch_dir = Path(directory)
    if not watch_dir.is_dir():
        print("error: ingest directory does not exist", file=sys.stderr)
        return 2

    try:
        thresholds = load_thresholds()
        session_factory = make_session_factory(
            ingest_config.data_dir / "imageguard.db"
        )
        analyses = AnalysisRepository(session_factory)
        service = AnalysisService(
            thresholds,
            ImageRepository(session_factory),
            analyses,
            ArtifactStore(ingest_config.data_dir / "artifacts"),
        )
        ingestor = FolderIngestor(
            ingest_config, service, thresholds.snapshot_hash()
        )
    except Exception as exc:  # noqa: BLE001 - keep terminal errors path-free
        print(f"error: ingest setup failed ({type(exc).__name__})", file=sys.stderr)
        return 1

    if mode == "once":
        try:
            summary = ingestor.scan_once(watch_dir)
        except Exception as exc:  # noqa: BLE001 - keep terminal errors path-free
            print(f"error: ingest scan failed ({type(exc).__name__})", file=sys.stderr)
            return 1
        _print_summary(mode, summary)
        return 0

    try:
        while True:
            summary = ingestor.scan_once(watch_dir)
            if summary.has_activity:
                _print_summary(mode, summary)
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - keep terminal errors path-free
        print(f"error: ingest watch failed ({type(exc).__name__})", file=sys.stderr)
        return 1


def _print_summary(mode: str, summary) -> None:
    print(
        f"ingest {mode}: analyzed={summary.analyzed} errors={summary.errors} "
        f"skipped_duplicate={summary.skipped_duplicate} deferred={summary.deferred}"
    )
