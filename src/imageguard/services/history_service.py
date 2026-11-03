"""HistoryService — list/get/delete analyses + session stats."""

from __future__ import annotations

from imageguard.storage.artifacts import ArtifactStore
from imageguard.storage.repository import AnalysisRepository


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
            } for m in a.method_results],
            "artifacts": [{"kind": x.kind, "method": x.method,
                           "rel_path": x.rel_path} for x in a.artifacts],
        }

    def delete(self, analysis_id: str) -> bool:
        rels = self.analyses.delete(analysis_id)
        if not rels and self.analyses.get(analysis_id) is None:
            pass  # row đã xóa hoặc không tồn tại
        self.store.delete_analysis(analysis_id)  # dọn file artifacts
        return True

    def stats(self) -> dict:
        return self.analyses.stats()
