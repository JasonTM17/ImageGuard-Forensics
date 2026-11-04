"""AnalysisService — orchestrate toàn pipeline cho một ảnh.

safe_open → methods (có progress callbacks) → fusion → persist (dedup
image theo sha256) → artifacts → ForensicReport. Entry point UI gọi qua
worker thread; hàm này Qt-free, chạy được headless.
"""

from __future__ import annotations

import json
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from imageguard import __version__
from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import MethodName, Verdict
from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.core.models import AlgorithmResult, ForensicReport, ImageDocument
from imageguard.forensics.auxiliary.meta_report import audit_exif
from imageguard.forensics.fusion import combine
from imageguard.forensics.registry import load_method
from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.repository import AnalysisRepository, ImageRepository

DEFAULT_METHODS = ["keypoint", "block", "ela"]
ProgressCB = Callable[[float, str], None] | None


@dataclass(slots=True)
class AnalysisRequest:
    path: Path
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    cancel: threading.Event | None = None
    progress_cb: ProgressCB = None
    include_exif: bool = True


class AnalysisService:
    def __init__(self, cfg: ThresholdSet | None = None,
                 images: ImageRepository | None = None,
                 analyses: AnalysisRepository | None = None,
                 store: ArtifactStore | None = None) -> None:
        self.cfg = cfg or load_thresholds()
        self.images = images
        self.analyses = analyses
        self.store = store or ArtifactStore()

    def analyze(self, req: AnalysisRequest) -> ForensicReport | ErrorResult:
        t0 = time.perf_counter()
        _progress(req.progress_cb, 0.05, "safe_open")
        doc = safe_open(req.path, self.cfg)
        if isinstance(doc, ErrorResult):
            return doc

        report_id = str(uuid.uuid4())
        results: list[AlgorithmResult] = []
        specs = list(req.methods)
        for i, spec in enumerate(specs):
            if req.cancel is not None and req.cancel.is_set():
                results.append(_cancelled_result(spec))
                break
            _progress(req.progress_cb, 0.1 + 0.7 * i / len(specs), spec)
            try:
                r = load_method(spec).analyze(doc, self.cfg, req.cancel)
            except Exception as e:
                r = _error_result(spec, e)
            results.append(r)
        if req.include_exif:
            results.append(audit_exif(doc, self.cfg))

        _progress(req.progress_cb, 0.85, "fusion")
        overall, signals, warnings = combine(results)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        report = _build_report(report_id, doc, results, overall, signals,
                               warnings, self.cfg, wall_ms)

        _progress(req.progress_cb, 0.9, "persist")
        if self.images is not None and self.analyses is not None:
            try:
                self._persist(report, doc, results, wall_ms)
            except Exception as e:
                # Phân tích đã tính xong — persistence lỗi KHÔNG được huỷ kết quả.
                # Báo trong limitations để report tự nói rõ mất persist.
                report.limitations.append(
                    f"lưu lịch sử/artifact thất bại — kết quả phân tích vẫn hợp lệ ({type(e).__name__}: {e})")

        _progress(req.progress_cb, 1.0, "done")
        return report

    def _persist(self, report: ForensicReport, doc: ImageDocument,
                 results: list[AlgorithmResult], wall_ms: int) -> None:
        asset = self.images.get_or_create(
            doc.sha256, doc.file_name, doc.width, doc.height, doc.image_format)
        rd = _report_dict(report)
        # duration wall-clock (toàn pipeline) — không phải tổng method duration.
        self.analyses.save_report(report.report_id, asset.id, rd,
                                  self.cfg.snapshot_hash(), wall_ms)
        for r in results:
            if r.mask is not None:
                rel = self.store.put_image(report.report_id,
                                           f"{r.method.value}_mask.png", r.mask)
                self.analyses.add_artifact(report.report_id, "mask",
                                           r.method.value, rel)
            if r.heatmap is not None:
                hm = (np.clip(r.heatmap, 0, 1) * 255).astype(np.uint8)
                rel = self.store.put_image(report.report_id,
                                           f"{r.method.value}_heat.png", hm)
                self.analyses.add_artifact(report.report_id, "heatmap",
                                           r.method.value, rel)
            if r.evidence.get("pairs_image") is not None:
                rel = self.store.put_image(report.report_id,
                                           f"{r.method.value}_pairs.png",
                                           r.evidence["pairs_image"])
                self.analyses.add_artifact(report.report_id, "pairs",
                                           r.method.value, rel)
        rel = self.store.put_bytes(
            report.report_id, "report.json",
            json.dumps(rd, ensure_ascii=False, default=str, indent=1).encode())
        self.analyses.add_artifact(report.report_id, "report", "", rel)


def _progress(cb: ProgressCB, frac: float, stage: str) -> None:
    if cb:
        cb(frac, stage)


def _env() -> dict[str, str]:
    import numpy
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": numpy.__version__}


# Registry spec → MethodName (registry dùng alias ngắn, MethodName dùng value đầy đủ).
# Trước đây các spec không phải MethodName value (keypoint, block, noise, freq,
# splicing) đều rơi về COPY_MOVE_KEYPOINT khi method crash/cancel — sai nhãn.
_SPEC_TO_METHOD: dict[str, MethodName] = {
    "keypoint": MethodName.COPY_MOVE_KEYPOINT,
    "block": MethodName.COPY_MOVE_BLOCK,
    "ela": MethodName.ELA,
    "noise": MethodName.NOISE_INCONSISTENCY,
    "noise_local": MethodName.NOISE_LOCAL,
    "freq": MethodName.FREQUENCY_ANALYSIS,
    "splicing": MethodName.SPLICING_CUES,
    "exif": MethodName.EXIF_AUDIT,
}


def _try_method_value(spec: str) -> MethodName:
    try:
        return MethodName(spec)
    except ValueError:
        return MethodName.COPY_MOVE_KEYPOINT


def _error_result(spec: str, e: Exception) -> AlgorithmResult:
    return AlgorithmResult(
        method=_SPEC_TO_METHOD.get(spec, _try_method_value(spec)),
        verdict=Verdict.ERROR, evidence={"spec": spec, "error": str(e)},
        limitations=["method crashed — xem evidence.error"])


def _cancelled_result(spec: str) -> AlgorithmResult:
    return AlgorithmResult(
        method=_SPEC_TO_METHOD.get(spec, _try_method_value(spec)),
        verdict=Verdict.ERROR, evidence={"spec": spec, "error": "cancelled"},
        limitations=["analysis bị huỷ trước khi method chạy"])


def _build_report(rid: str, doc: ImageDocument,
                  results: list[AlgorithmResult], overall: Verdict,
                  signals: list[str], warnings: list[str],
                  cfg: ThresholdSet, dur_ms: int) -> ForensicReport:
    limitations = list(dict.fromkeys(
        lim for r in results for lim in r.limitations))
    limitations.extend(warnings)
    return ForensicReport(
        report_id=rid, image_sha256=doc.sha256, image_name=doc.file_name,
        image_width=doc.width, image_height=doc.height,
        created_at=datetime.now(timezone.utc).isoformat(),
        tool_version=__version__, env=_env(),
        thresholds_hash=cfg.snapshot_hash(), methods=results,
        overall_verdict=overall, limitations=limitations)


def report_dur_ms(report: ForensicReport) -> int:
    return sum(m.duration_ms for m in report.methods)


def _report_dict(report: ForensicReport) -> dict:
    return {
        "report_id": report.report_id,
        "image_sha256": report.image_sha256,
        "image_name": report.image_name,
        "image_width": report.image_width, "image_height": report.image_height,
        "created_at": report.created_at, "tool_version": report.tool_version,
        "env": report.env, "thresholds_hash": report.thresholds_hash,
        "overall_verdict": report.overall_verdict.value,
        "limitations": report.limitations, "disclaimer": report.disclaimer,
        "methods": [{
            "method": m.method.value, "verdict": m.verdict.value,
            "score": m.score, "score_definition": m.score_definition,
            "evidence": {k: v for k, v in m.evidence.items()
                         if not isinstance(v, np.ndarray)},
            "limitations": m.limitations,
            "params_hash": m.params_hash, "duration_ms": m.duration_ms,
            "has_mask": m.mask is not None,
        } for m in report.methods],
    }
