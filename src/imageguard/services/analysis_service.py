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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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


class _PersistRecoveryError(Exception):
    """Persist fail VÀ cleanup không hoàn tất — KHÔNG phải save bình thường.

    Chỉ mang report UUID + stage lỗi + loại exception — không đường dẫn
    tuyệt đối, không raw filesystem/DB message (nằm trong ``__cause__``).
    """

    def __init__(self, report_id: str, failed_stages: list[str]) -> None:
        self.report_id = report_id
        self.failed_stages = tuple(failed_stages)
        super().__init__(
            f"cleanup persist không hoàn tất — report {report_id} cần "
            f"recovery thủ công (stages lỗi: {', '.join(failed_stages)})")


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
            except Exception as e:  # noqa: BLE001 - method isolation: method crash/cancel không được huỷ pipeline
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
            except _PersistRecoveryError as e:
                # str() của recovery error được thiết kế an toàn: chỉ UUID +
                # stage + loại exception — không đường dẫn, không raw message.
                report.limitations.append(
                    "lưu lịch sử/artifact thất bại — kết quả phân tích vẫn "
                    f"hợp lệ ({type(e).__name__}: {e})")
            except Exception as e:  # noqa: BLE001 - ranh giới persistence: lỗi DB/artifact không được huỷ kết quả phân tích
                # Exception generic có thể chứa đường dẫn tuyệt đối/host path
                # trong str() — chỉ giữ tên loại, đúng floor không lộ host path.
                report.limitations.append(
                    "lưu lịch sử/artifact thất bại — kết quả phân tích vẫn "
                    f"hợp lệ ({type(e).__name__})")

        _progress(req.progress_cb, 1.0, "done")
        return report

    def _persist(self, report: ForensicReport, doc: ImageDocument,
                 results: list[AlgorithmResult], wall_ms: int) -> None:
        rid = report.report_id
        self._claim_report_id(rid)
        try:
            asset = self.images.get_or_create(
                doc.sha256, doc.file_name, doc.width, doc.height,
                doc.image_format)
            rd = _report_dict(report)
            # duration wall-clock (toàn pipeline) — không phải tổng method duration.
            self.analyses.save_report(rid, asset.id, rd,
                                      self.cfg.snapshot_hash(), wall_ms)
            for r in results:
                if r.mask is not None:
                    rel = self.store.put_image(rid,
                                               f"{r.method.value}_mask.png",
                                               r.mask)
                    self.analyses.add_artifact(rid, "mask",
                                               r.method.value, rel)
                if r.heatmap is not None:
                    clean_hm = np.nan_to_num(
                        np.asarray(r.heatmap, dtype=np.float32),
                        nan=0.0, posinf=1.0, neginf=0.0,
                    )
                    hm = (np.clip(clean_hm, 0.0, 1.0) * 255).astype(np.uint8)
                    rel = self.store.put_image(rid,
                                               f"{r.method.value}_heat.png",
                                               hm)
                    self.analyses.add_artifact(rid, "heatmap",
                                               r.method.value, rel)
                if r.evidence.get("pairs_image") is not None:
                    rel = self.store.put_image(rid,
                                               f"{r.method.value}_pairs.png",
                                               r.evidence["pairs_image"])
                    self.analyses.add_artifact(rid, "pairs",
                                               r.method.value, rel)
            rel = self.store.put_bytes(
                rid, "report.json",
                json.dumps(rd, ensure_ascii=False, default=str,
                           indent=1).encode())
            self.analyses.add_artifact(rid, "report", "", rel)
        except Exception as exc:  # noqa: BLE001 - compensation boundary: mọi persist fail thường đều phải dọn custody dở dang
            self._compensate_owned(rid, exc)

    def _claim_report_id(self, rid: str) -> None:
        """Giành ownership độc quyền cho report UUID MỚI trước khi persist.

        Reject khi đã có analysis row hoặc artifact dir — raise trước mọi ghi
        nên compensation không bao giờ xoá dữ liệu có sẵn.
        """
        if self.analyses.get(rid) is not None:
            raise ValueError(f"report id đã tồn tại trong lịch sử: {rid}")
        self.store._reserve_dir(rid)

    def _compensate_owned(self, rid: str, exc: Exception) -> None:
        """Thu hồi persist dở dang của UUID đã claim — mỗi nhánh độc lập.

        Cleanup sạch → re-raise lỗi gốc (analyze báo limitation như cũ).
        Cleanup fail → _PersistRecoveryError nêu rõ recovery, không giả save
        thường. Không đụng ImageAsset/source bytes/report khác.
        """
        failed: list[str] = []
        try:
            self.analyses.delete(rid)
        except Exception as e:  # noqa: BLE001 - vẫn phải thử cleanup artifacts
            failed.append(f"database:{type(e).__name__}")
        try:
            self.store.delete_analysis(rid)
        except Exception as e:  # noqa: BLE001
            failed.append(f"artifacts:{type(e).__name__}")
        if failed:
            raise _PersistRecoveryError(rid, failed) from exc
        raise exc


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
    "deepfake": MethodName.DEEPFAKE_CUES,
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
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__, env=_env(),
        thresholds_hash=cfg.snapshot_hash(), methods=results,
        overall_verdict=overall, limitations=limitations)


def report_dur_ms(report: ForensicReport) -> int:
    return sum(m.duration_ms for m in report.methods)


def report_to_dict(report: ForensicReport) -> dict:
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


def reclaim_orphan_artifacts(
    store: ArtifactStore | None, analyses: AnalysisRepository | None
) -> tuple[list[str], list[str]]:
    """Startup sweep dir artifact mồ côi (dir có, row Analysis không có).

    Fail-closed theo hai tầng: đọc danh sách id lỗi → trả rỗng và KHÔNG xóa
    gì; dir nào không xóa được (file bị khóa) được bỏ qua cho lần sau. Caller:
    app boot, CLI analyze persist, ingest setup.
    """
    if store is None or analyses is None:
        return [], []
    try:
        known = analyses.all_ids()
    except Exception:  # noqa: BLE001 - sweep phải fail-closed: DB lỗi = không xóa
        return [], []
    return store.sweep_orphan_report_dirs(known)


_report_dict = report_to_dict
