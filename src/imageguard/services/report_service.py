"""ReportService — đọc report.json + export Markdown.

Markdown-only theo cut-list P08 (PDF backend có thể thêm sau — report.md
đã đủ cho capstone/demo). Report chứa: metadata, thresholds hash,
per-method cards, limitations, disclaimer, provenance — KHÔNG xác suất.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from imageguard.core.paths import data_root
from imageguard.storage.artifacts import ArtifactStore
from imageguard.ui.strings import VERDICT_LABEL

DISCLAIMER_MD = (
    "> Công cụ chỉ phát hiện dấu hiệu nghi vấn từ phân tích tự động — "
    "không khẳng định ảnh bị can thiệp, cũng không xác thực nguyên bản. "
    "ELA/EXIF/noise là bằng chứng khám phá, không tự lập thành kết luận."
)


class ReportService:
    def __init__(self, store: ArtifactStore | None = None) -> None:
        self.store = store or ArtifactStore()

    def read_report_json(self, analysis_id: str) -> dict | None:
        try:
            report = json.loads(self.store.read(f"{analysis_id}/report.json"))
        except (ValueError, OSError):
            return None
        return report if isinstance(report, dict) else None

    def export_markdown(self, analysis_id: str,
                        out_path: Path | None = None) -> Path | None:
        """Render report.md từ report.json artifact. Trả path hoặc None."""
        rep = self.read_report_json(analysis_id)
        if rep is None:
            return None
        canonical = self.store.root.resolve() / self.store.rel_path(analysis_id, "report.md")
        self._guard_export_destination(canonical, canonical)
        if out_path is not None:
            out_path = Path(out_path)
            self._guard_export_destination(out_path, canonical)
        data = self._render(rep).encode("utf-8")
        self.store.put_bytes(analysis_id, "report.md", data)
        if out_path is not None:
            if out_path.resolve() != canonical:
                self._write_external(out_path, data, canonical)
            return out_path
        return self.store.abs_path(f"{analysis_id}/report.md")

    def _guard_export_destination(self, destination: Path, canonical: Path) -> None:
        target = destination.resolve()
        artifacts_root = self.store.root.resolve()
        if (canonical.resolve() != canonical
                or (target.is_relative_to(artifacts_root)
                    and target != canonical)):
            raise ValueError("export destination would replace a stored artifact")
        # Vùng data runtime (SQLite, backup, file nội bộ) nằm NGOÀI artifact
        # root: kê từ chỉ che artifacts subtree — chặn cả phần còn lại của
        # data_root để export không ghi đè dữ liệu vận hành khác.
        data = data_root().resolve()
        if target.is_relative_to(data) and not target.is_relative_to(artifacts_root):
            raise ValueError(
                "export destination nằm trong thư mục data runtime "
                "— chọn nơi khác")

    def _write_external(self, destination: Path, data: bytes, canonical: Path) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".report-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self._guard_export_destination(destination, canonical)
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _render(self, rep: dict) -> str:
        v = rep.get("overall_verdict", "inconclusive")
        env = rep.get("env", {})
        lines = [
            f"# Báo cáo phân tích — {rep.get('image_name', '?')}",
            "",
            DISCLAIMER_MD,
            "",
            "## Tổng quan",
            "",
            "| Trường | Giá trị |",
            "|---|---|",
            f"| Verdict tổng | **{VERDICT_LABEL.get(v, v)}** |",
            f"| Report ID | `{rep.get('report_id', '')}` |",
            (f"| Ảnh | {rep.get('image_name')} ({rep.get('image_width')}×"
             f"{rep.get('image_height')}) |"),
            f"| SHA-256 | `{rep.get('image_sha256', '')[:16]}…` |",
            f"| Thời gian | {rep.get('created_at', '')[:19]}Z |",
            f"| Tool | ImageGuard {rep.get('tool_version', '?')} |",
            (f"| Env | python {env.get('python', '?')} · opencv "
             f"{env.get('opencv', '?')} · numpy {env.get('numpy', '?')} |"),
            f"| Thresholds hash | `{rep.get('thresholds_hash', '')}` |",
            "",
            "## Kết quả theo phương pháp",
            "",
        ]
        for m in rep.get("methods", []):
            mv = m.get("verdict", "?")
            lines += [
                f"### {m.get('method', '?')} — {VERDICT_LABEL.get(mv, mv)}",
                "",
            ]
            if m.get("score") is not None:
                lines.append(f"- **score** = {m['score']:.4g} "
                             f"({m.get('score_definition', '')})")
            lines.append(f"- duration: {m.get('duration_ms', 0)} ms"
                         + (", có mask" if m.get("has_mask") else ""))
            for k, val in (m.get("evidence") or {}).items():
                lines.append(f"- {k}: {val}")
            for lim in m.get("limitations") or []:
                lines.append(f"- ⚠ {lim}")
            lines.append("")
        lims = rep.get("limitations") or []
        if lims:
            lines += ["## Cảnh báo tổng", ""]
            lines += [f"- {lim}" for lim in lims]
            lines.append("")
        lines += [
            "---",
            (f"Xuất lúc {datetime.now(UTC).isoformat(timespec='seconds')}"
             " — report tái lập được từ artifact, không chứa xác suất."),
        ]
        return "\n".join(lines) + "\n"
