"""Serial folder polling for stable, finished still images."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from imageguard import __version__
from imageguard.core.config import load_thresholds
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.limits import MAGIC_TABLE, SNIFF_BYTES
from imageguard.enhancement.derive import derive, write_sibling
from imageguard.ingest.config import IngestConfig
from imageguard.ingest.custody import CustodyLog
from imageguard.services.analysis_service import AnalysisRequest, AnalysisService

_IGNORED_SUFFIXES = (".tmp", ".partial", ".filepart")
_COMPLETED_EVENTS = {"analyzed", "skipped_duplicate"}


@dataclass(slots=True)
class ScanSummary:
    analyzed: int = 0
    errors: int = 0
    skipped_duplicate: int = 0
    deferred: int = 0

    @property
    def has_activity(self) -> bool:
        return any((self.analyzed, self.errors, self.skipped_duplicate, self.deferred))


class FolderIngestor:
    """Scan a folder serially, leaving files on disk after analysis."""

    def __init__(
        self,
        config: IngestConfig,
        service: AnalysisService,
        config_hash: str,
        *,
        settle_delay: float = 0.1,
    ) -> None:
        if not 0 <= settle_delay < 0.5:
            raise ValueError("settle_delay must be between zero and 0.5 seconds")
        self.config = config
        self.service = service
        self.config_hash = config_hash
        self.settle_delay = settle_delay
        self.custody = CustodyLog(config.custody_path)
        self._latest_events = self.custody.latest_events()
        self._seen_hashes: set[str] = set()
        self._handled: dict[Path, tuple[int, int]] = {}

    def scan_once(self, directory: Path | str) -> ScanSummary:
        watch_dir = Path(directory)
        if not watch_dir.is_dir():
            raise NotADirectoryError("ingest directory does not exist")

        self._pull_due(watch_dir)
        candidates = []
        for path in sorted(watch_dir.iterdir(), key=lambda item: item.name.casefold()):
            if self._ignored(path):
                continue
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            signature = self._stable_signature(path)
            if signature is None or self._handled.get(path) == signature:
                continue
            if not _has_supported_magic(path):
                self._handled[path] = signature
                continue
            try:
                image_sha256 = _sha256_file(path)
            except OSError:
                continue
            if _file_signature(path) != signature:
                continue

            key = (path.name, image_sha256)
            if self._latest_events.get(key) in _COMPLETED_EVENTS:
                self._handled[path] = signature
                continue
            candidates.append((path, signature, image_sha256, key))

        candidates.sort(
            key=lambda item: (
                self._latest_events.get(item[3]) != "deferred",
                item[0].name.casefold(),
            )
        )
        summary = ScanSummary()
        submitted = 0
        for path, signature, image_sha256, key in candidates:
            if image_sha256 in self._seen_hashes:
                self._append_event(
                    "skipped_duplicate", path, signature, image_sha256,
                    analysis_id=None,
                )
                self._handled[path] = signature
                summary.skipped_duplicate += 1
                continue

            if submitted >= self.config.max_pending:
                if self._latest_events.get(key) != "deferred":
                    self._append_event(
                        "deferred", path, signature, image_sha256,
                        analysis_id=None,
                    )
                    summary.deferred += 1
                continue

            submitted += 1
            self._seen_hashes.add(image_sha256)
            try:
                result = self.service.analyze(
                    AnalysisRequest(path=path, methods=list(self.config.methods))
                )
            except Exception:  # noqa: BLE001 - isolate failures to one folder item
                self._append_event(
                    "error", path, signature, image_sha256,
                    analysis_id=None, error_code="INTERNAL",
                )
                self._handled[path] = signature
                summary.errors += 1
                continue

            if isinstance(result, ErrorResult):
                self._append_event(
                    "error", path, signature, image_sha256,
                    analysis_id=None, error_code=result.code.value,
                )
                summary.errors += 1
            else:
                sibling_error: str | None = None
                if self.config.enhance_sibling:
                    sibling_error = self._try_enhance_sibling(path, image_sha256)
                self._append_event(
                    "analyzed", path, signature, image_sha256,
                    analysis_id=result.report_id,
                    error_code=sibling_error,
                )
                summary.analyzed += 1
            self._handled[path] = signature

        return summary

    def _stable_signature(self, path: Path) -> tuple[int, int] | None:
        first = _file_signature(path)
        if first is None:
            return None
        if self.settle_delay:
            time.sleep(self.settle_delay)
        second = _file_signature(path)
        return second if first == second else None

    def _ignored(self, path: Path) -> bool:
        name = path.name
        if name.startswith(".") or name.casefold().endswith(_IGNORED_SUFFIXES):
            return True
        stem = path.stem.casefold()
        return stem.endswith(".view") and path.suffix.casefold() in (".png", ".json")

    def _try_enhance_sibling(self, path: Path, image_sha256: str) -> str | None:
        """Write ``<stem>.view.png`` (+ .json) next to source; return error tag or None."""
        params = {"ksize": 5, "sigma": 1.0, "amount": 0.7}
        cfg = getattr(self.service, "cfg", None) or load_thresholds()
        doc = safe_open(path, cfg)
        if isinstance(doc, ErrorResult):
            return "enhance_sibling_failed"
        try:
            out, _meta = derive(doc.array, "unsharp", params)
            dest_png = path.with_name(path.stem + ".view.png")
            write_sibling(
                path, dest_png, out, "unsharp", params, image_sha256
            )
        except (OSError, ValueError):
            return "enhance_sibling_failed"
        except Exception:  # noqa: BLE001 — do not fail scan
            return "enhance_sibling_failed"
        return None

    def _append_event(
        self,
        event: str,
        path: Path,
        signature: tuple[int, int],
        image_sha256: str,
        *,
        analysis_id: str | None,
        error_code: str | None = None,
    ) -> None:
        self.custody.append(
            event=event,
            image_sha256=image_sha256,
            basename=path.name,
            camera_label=self.config.camera_label,
            file_mtime=_mtime_utc(signature[1]),
            analysis_id=analysis_id,
            config_hash=self.config_hash,
            tool_version=__version__,
            consent_note=self.config.consent_note,
            error_code=error_code,
        )
        self._latest_events[(path.name, image_sha256)] = event

    def _pull_due(self, watch_dir: Path) -> None:
        http = self.config.http
        if not http or http.get("enabled") is not True:
            return
        try:
            from imageguard.ingest.http_pull import pull_due
        except ImportError:
            return
        pull_due(http, watch_dir)


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _has_supported_magic(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            head = stream.read(SNIFF_BYTES)
    except OSError:
        return False
    for magic, image_format in MAGIC_TABLE:
        if head.startswith(magic):
            return image_format != "webp" or head[8:12] == b"WEBP"
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtime_utc(mtime_ns: int) -> str:
    try:
        return datetime.fromtimestamp(mtime_ns / 1_000_000_000, UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return str(mtime_ns)
