"""Repositories — parameterized queries only (SQLAlchemy ORM)."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from imageguard.storage.models import (
    Analysis,
    Artifact,
    Experiment,
    ExperimentResult,
    ImageAsset,
    MethodResult,
)


class ImageRepository:
    def __init__(self, sf: sessionmaker[Session]) -> None:
        self._sf = sf

    def get_or_create(self, sha256: str, file_name: str, w: int, h: int,
                      fmt: str) -> ImageAsset:
        with self._sf() as s:
            row = s.scalar(select(ImageAsset).where(ImageAsset.sha256 == sha256))
            if row:
                return row
            row = ImageAsset(sha256=sha256, file_name=file_name, width=w,
                             height=h, format=fmt)
            s.add(row)
            s.commit()
            return row


class AnalysisRepository:
    def __init__(self, sf: sessionmaker[Session]) -> None:
        self._sf = sf

    def save_report(self, analysis_id: str, image_id: int, report_dict: dict,
                    config_hash: str, duration_ms: int) -> None:
        with self._sf() as s:
            a = Analysis(
                id=analysis_id, image_id=image_id,
                tool_version=report_dict["tool_version"],
                overall_verdict=report_dict["overall_verdict"],
                config_hash=config_hash, duration_ms=duration_ms,
                limitations_json=json.dumps(report_dict["limitations"],
                                            ensure_ascii=False),
            )
            s.add(a)
            for m in report_dict["methods"]:
                s.add(MethodResult(
                    analysis_id=analysis_id, method=m["method"],
                    verdict=m["verdict"], score=m["score"],
                    score_definition=m["score_definition"],
                    evidence_json=json.dumps(m["evidence"], ensure_ascii=False,
                                             default=str),
                    limitations_json=json.dumps(m["limitations"],
                                                ensure_ascii=False),
                    duration_ms=m["duration_ms"]))
            s.commit()

    def add_artifact(self, analysis_id: str, kind: str, method: str,
                     rel_path: str) -> None:
        _assert_relative(rel_path)
        with self._sf() as s:
            s.add(Artifact(analysis_id=analysis_id, kind=kind, method=method,
                           rel_path=rel_path))
            s.commit()

    def list(self, limit: int = 100) -> list[Analysis]:
        # Eager load — trả object detached ra ngoài session, lazy load sẽ crash.
        with self._sf() as s:
            return list(s.scalars(
                select(Analysis)
                .options(selectinload(Analysis.image),
                         selectinload(Analysis.method_results))
                .order_by(Analysis.created_at.desc())
                .limit(limit)))

    def get(self, analysis_id: str) -> Analysis | None:
        with self._sf() as s:
            return s.scalar(
                select(Analysis)
                .options(selectinload(Analysis.image),
                         selectinload(Analysis.method_results),
                         selectinload(Analysis.artifacts))
                .where(Analysis.id == analysis_id))

    def delete(self, analysis_id: str) -> list[str]:
        """Xóa row; trả rel_paths artifact để caller dọn file."""
        with self._sf() as s:
            a = s.get(Analysis, analysis_id)
            if a is None:
                return []
            rels = [art.rel_path for art in a.artifacts]
            s.delete(a)  # cascade xóa method_results + artifacts rows
            s.commit()
            return rels

    def stats(self) -> dict:
        with self._sf() as s:
            total = s.scalar(select(func.count(Analysis.id))) or 0
            by_verdict = dict(s.execute(
                select(Analysis.overall_verdict, func.count())
                .group_by(Analysis.overall_verdict)).all())
            avg_ms = s.scalar(select(func.avg(Analysis.duration_ms)))
            return {"n_analyses": total, "by_verdict": by_verdict,
                    "avg_duration_ms": float(avg_ms) if avg_ms else None,
                    "label": "thống kê phiên — không phải benchmark"}


class ExperimentRepository:
    def __init__(self, sf: sessionmaker[Session]) -> None:
        self._sf = sf

    def save(self, report_dict: dict) -> None:
        with self._sf() as s:
            e = Experiment(
                experiment_id=report_dict["experiment_id"],
                dataset=report_dict["dataset"], split=report_dict["split"],
                provenance_json=json.dumps(report_dict["provenance"],
                                           default=str))
            s.add(e)
            s.flush()
            for method, metrics in report_dict["per_method"].items():
                s.add(ExperimentResult(
                    experiment_id=e.id, method=method,
                    metrics_json=json.dumps(metrics, default=str)))
            s.commit()


def _assert_relative(rel: str) -> None:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"DB chỉ lưu relative path an toàn: {rel!r}")
