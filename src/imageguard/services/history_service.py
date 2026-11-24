"""HistoryService — list/get/delete analyses + session stats."""

from __future__ import annotations

import json

from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.repository import AnalysisRepository


def _saved_json(value: str, empty: dict | list) -> dict | list:
    """Legacy nullable or malformed fields must remain renderable in history."""
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return empty
    return decoded if isinstance(decoded, type(empty)) else empty


class HistoryService:
    def __init__(self, analyses: AnalysisRepository,
                 store: ArtifactStore | None = None) -> None:
        self.analyses = analyses
        self.store = store or ArtifactStore()

    def list(self, limit: int = 100) -> list[dict]:
        return [{
            "id": a.id, "created_at": str(a.created_at),
            "verdict": a.overall_verdict, "duration_ms": a.duration_ms,
            "image_name": a.image.file_name if a.image else "?",
            "n_methods": len(a.method_results),
        } for a in self.analyses.list(limit)]

    def get(self, analysis_id: str) -> dict | None:
        a = self.analyses.get(analysis_id)
        if a is None:
            return None
        return {
            "id": a.id, "created_at": str(a.created_at),
            "verdict": a.overall_verdict, "duration_ms": a.duration_ms,
            "image_name": a.image.file_name if a.image else "?",
            "config_hash": a.config_hash,
            "methods": [{
                "method": m.method, "verdict": m.verdict, "score": m.score,
                "score_definition": m.score_definition,
                "evidence": _saved_json(m.evidence_json, {}),
                "limitations": _saved_json(m.limitations_json, []),
                "duration_ms": m.duration_ms,
            } for m in a.method_results],
            "artifacts": [{"kind": x.kind, "method": x.method,
                           "rel_path": x.rel_path} for x in a.artifacts],
        }

    def delete(self, analysis_id: str) -> bool:
        existed = self.analyses.get(analysis_id) is not None
        # Dọn file artifacts TRƯỚC khi xóa row: file bị khóa trên Windows
        # (AV/viewer giữ handle) sẽ raise trước khi DB commit — row còn
        # nguyên để lịch sử hiển thị và người dùng retry, thay vì để lại
        # artifact mồ côi không còn row nào trỏ tới.
        self.store.delete_analysis(analysis_id)
        if existed:
            self.analyses.delete(analysis_id)
        return True

    def stats(self) -> dict:
        return self.analyses.stats()
