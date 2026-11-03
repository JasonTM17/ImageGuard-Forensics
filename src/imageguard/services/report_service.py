"""ReportService — stub P06; export đầy đủ (HTML/PDF) ở P08.

Hiện tại: report.json artifact đã được AnalysisService ghi — service này
chỉ expose điểm đọc để UI không phụ thuộc storage layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from imageguard.storage.artifacts import ArtifactStore


class ReportService:
    def __init__(self, store: ArtifactStore | None = None) -> None:
        self.store = store or ArtifactStore()

    def read_report_json(self, analysis_id: str) -> dict | None:
        try:
            return json.loads(self.store.read(f"{analysis_id}/report.json"))
        except Exception:
            return None
