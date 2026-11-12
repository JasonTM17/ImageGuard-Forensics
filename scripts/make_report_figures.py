"""Generate reproducible synthetic figures for the course report (no real photos).

Outputs PNGs under ``docs/report/figures/`` and is safe to re-run (fixed seeds).
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Offscreen Qt must be set before PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "report" / "figures"
SEED = 20261112
GUI_SIZE = (1280, 800)
MAX_PNG_BYTES = 300 * 1024


@dataclass(frozen=True, slots=True)
class FigureRecord:
    """One generated asset for figures.md."""

    filename: str
    title_vi: str
    desc_vi: str
    producer: str


def _validate_out_dir(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("output directory must live inside the repository")
    return resolved


def write_png_limited(bgr: np.ndarray, path: Path, max_bytes: int = MAX_PNG_BYTES) -> int:
    """Encode BGR to PNG, downscaling until ``max_bytes`` if needed."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("image must be H×W×3 BGR")
    if bgr.size == 0:
        raise ValueError("image must be non-empty")
    source = np.ascontiguousarray(bgr)
    last_size = 0
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.28):
        if scale != 1.0:
            h, w = source.shape[:2]
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = cv2.resize(source, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            img = source
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError("OpenCV could not encode PNG")
        payload = buf.tobytes()
        last_size = len(payload)
        if last_size <= max_bytes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return last_size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return last_size


def _labeled_row(panels: list[np.ndarray], labels: list[str], gap: int = 8) -> np.ndarray:
    """Stack equal-height panels with a short Vietnamese caption row."""
    if not panels or len(panels) != len(labels):
        raise ValueError("panels and labels must be non-empty and equal length")
    label_px = 28
    heights = [p.shape[0] for p in panels]
    target_h = max(heights)
    normed: list[np.ndarray] = []
    for p in panels:
        if p.shape[0] != target_h:
            w = max(1, int(p.shape[1] * target_h / p.shape[0]))
            p = cv2.resize(p, (w, target_h), interpolation=cv2.INTER_AREA)
        normed.append(p)
    widths = [p.shape[1] for p in normed]
    total_w = sum(widths) + gap * (len(normed) - 1)
    canvas = np.full((target_h + label_px, total_w, 3), 245, np.uint8)
    x = 0
    for p, text, w in zip(normed, labels, widths, strict=True):
        h = p.shape[0]
        canvas[label_px : label_px + h, x : x + w] = p
        cv2.putText(
            canvas,
            text,
            (x + 6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        x += w + gap
    return canvas


def _synthetic_copymove_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from imageguard.evaluation.synthforge import ForgeSpec, _base_image, forge

    size = (512, 512)
    base = _base_image(rng, size)
    spec = ForgeSpec(
        size,
        "rect",
        [[120, 120], [280, 120], [280, 280], [120, 280]],
        (150, 150),
        0.0,
        1.0,
        "none",
        "easy",
    )
    forged, gt_mask = forge(base, spec)
    if not (gt_mask > 0).any():
        raise RuntimeError("synthetic ground-truth mask is empty")
    return base, forged, gt_mask


def _doc_from(bgr: np.ndarray, name: str = "synthetic.png") -> object:
    from imageguard.core.models import ImageDocument

    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="s" * 64,
        file_bytes=int(bgr.nbytes),
        file_name=name,
    )


def make_pipeline_figures(out_dir: Path, records: list[FigureRecord]) -> None:
    from imageguard.core.config import load_thresholds
    from imageguard.enhancement.derive import derive
    from imageguard.evaluation.enhance_study import RESTORER_PARAMS
    from imageguard.evaluation.synthforge import _base_image
    from imageguard.forensics.auxiliary.freq_view import FreqView
    from imageguard.forensics.block.pipeline import BlockCopyMove
    from imageguard.forensics.keypoint.pipeline import KeypointCopyMove
    from imageguard.forensics.keypoint.viz import overlay_mask
    from imageguard.forensics.localize.heatmap import overlay_heatmap, render_heatmap

    cfg = load_thresholds()
    rng = np.random.default_rng(SEED)

    original, forged, _gt = _synthetic_copymove_pair(rng)
    doc = _doc_from(forged)
    kp = KeypointCopyMove().analyze(doc, cfg)
    pairs_img = kp.evidence.get("pairs_image")
    if pairs_img is None or kp.mask is None:
        raise RuntimeError("keypoint pipeline did not produce pairs/mask on synthetic forge")
    mask_panel = overlay_mask(forged, kp.mask)
    row1 = _labeled_row(
        [original, forged],
        ["Anh goc (synthetic)", "Anh forge (synthetic)"],
        gap=12,
    )
    row2 = _labeled_row(
        [pairs_img, mask_panel],
        ["Cap keypoint (pipeline)", "Mask overlay (pipeline)"],
        gap=12,
    )
    copy_path = out_dir / "copymove_synthetic_keypoint.png"
    w = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] != w:
        row1 = cv2.copyMakeBorder(
            row1, 0, 0, 0, w - row1.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)
        )
    if row2.shape[1] != w:
        row2 = cv2.copyMakeBorder(
            row2, 0, 0, 0, w - row2.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245)
        )
    merged = np.vstack([row1, np.full((8, w, 3), 245, np.uint8), row2])
    size_b = write_png_limited(merged, copy_path)
    records.append(
        FigureRecord(
            "copymove_synthetic_keypoint.png",
            "Copy-move synthetic — keypoint",
            "So sánh ảnh gốc và ảnh forge synthetic; cặp keypoint và mask từ "
            "KeypointCopyMove (pipeline thật, seed cố định).",
            f"synthforge + KeypointCopyMove; PNG {size_b} byte",
        )
    )

    block = BlockCopyMove().analyze(doc, cfg)
    freq = FreqView().analyze(doc, cfg)
    if freq.heatmap is None:
        raise RuntimeError("FreqView did not return spectrum heatmap")
    block_vis = (
        overlay_heatmap(forged, block.heatmap)
        if block.heatmap is not None
        else forged
    )
    spec_vis = render_heatmap(freq.heatmap)
    panels = _labeled_row(
        [forged, block_vis, spec_vis],
        ["Forge", "Block-DCT heat", "FFT spectrum"],
        gap=10,
    )
    freq_path = out_dir / "freq_block_spectrum.png"
    size_f = write_png_limited(panels, freq_path)
    records.append(
        FigureRecord(
            "freq_block_spectrum.png",
            "Block-DCT và phổ tần số",
            "Ví dụ cue khám phá: heatmap block-DCT và phổ FFT trên cùng ảnh "
            "synthetic forge (không khẳng định độ chính xác).",
            f"BlockCopyMove + FreqView; PNG {size_f} byte",
        )
    )

    blur_src = _base_image(np.random.default_rng(SEED + 3), (420, 420))
    blurred = cv2.GaussianBlur(blur_src, (0, 0), sigmaX=2.8)
    unsharp, _ = derive(blurred, "unsharp", dict(RESTORER_PARAMS["unsharp"]))
    clahe, _ = derive(blurred, "clahe", dict(RESTORER_PARAMS["clahe"]))
    enh_panels = _labeled_row(
        [blurred, unsharp, clahe],
        ["Lam mo (synthetic)", "Unsharp", "CLAHE"],
        gap=10,
    )
    enh_path = out_dir / "enhance_synthetic_blur_compare.png"
    size_e = write_png_limited(enh_panels, enh_path)
    records.append(
        FigureRecord(
            "enhance_synthetic_blur_compare.png",
            "Tăng cường xem thử — blur synthetic",
            "Ba khung: ảnh synthetic bị Gaussian blur, unsharp mask và CLAHE "
            "(derive thật). Chỉ minh họa giao diện xem, không phải bằng chứng.",
            f"numpy blur + derive; PNG {size_e} byte",
        )
    )


def _qpixmap_to_bgr(pix) -> np.ndarray:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    if not buf.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError("QBuffer open failed")
    if not pix.save(buf, "PNG"):
        raise RuntimeError("QPixmap save failed")
    data = np.frombuffer(ba.data(), dtype=np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("could not decode grabbed PNG")
    return decoded


def _grab_main_window(window, filename: str, out_dir: Path) -> int:
    window.resize(*GUI_SIZE)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    bgr = _qpixmap_to_bgr(window.grab())
    return write_png_limited(bgr, out_dir / filename)


def _wait_worker(pool, worker, app, timeout_ms: int = 120_000) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)

    def finish() -> None:
        if loop.isRunning():
            loop.quit()

    worker.signals.result.connect(lambda *_: finish())
    worker.signals.failed.connect(lambda *_: finish())
    finished = getattr(worker.signals, "finished", None)
    if finished is not None:
        finished.connect(lambda: finish())
    timer.timeout.connect(finish)
    timer.start(timeout_ms)
    pool.start(worker)
    loop.exec()
    app.processEvents()
    if timer.isActive():
        timer.stop()
    else:
        raise TimeoutError("Qt worker timed out")


def make_gui_figures(out_dir: Path, forged_path: Path, records: list[FigureRecord]) -> None:
    from PySide6.QtWidgets import QApplication

    from imageguard.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    with tempfile.TemporaryDirectory(
        prefix="ig_report_fig_",
        ignore_cleanup_errors=True,
    ) as tmp:
        tmp_path = Path(tmp)
        analysis, history = __import__(
            "imageguard.ui.app",
            fromlist=["build_services"],
        ).build_services(
            db_path=tmp_path / "session.db",
            artifacts_root=tmp_path / "artifacts",
        )
        window = MainWindow(analysis, history)
        window.show()
        app.processEvents()
        try:
            _capture_gui_tabs(
                window, analysis, forged_path, tmp_path, out_dir, records, app
            )
        finally:
            window.close()
            del window, analysis, history
            gc.collect()
            time.sleep(0.15)
            app.processEvents()


def _capture_gui_tabs(
    window: object,
    analysis: object,
    forged_path: Path,
    tmp_path: Path,
    out_dir: Path,
    records: list[FigureRecord],
    app: object,
) -> None:
    from imageguard.evaluation.enhance_study import RESTORER_PARAMS
    from imageguard.evaluation.synthforge import _base_image
    from imageguard.services.analysis_service import AnalysisRequest
    from imageguard.ui.workers import AnalysisWorker, EnhanceWorker, make_pool

    analyze = window.analyze
    for name in (
        "chk_kp",
        "chk_bl",
        "chk_ela",
        "chk_noise",
        "chk_nloc",
        "chk_freq",
        "chk_spl",
    ):
        getattr(analyze, name).setChecked(True)
    analyze.load_path(forged_path)
    pool = make_pool()
    req = AnalysisRequest(path=forged_path, methods=analyze._methods())
    worker = AnalysisWorker(analysis, req)
    analyze._worker = worker
    worker.signals.progress.connect(analyze._on_progress)
    worker.signals.result.connect(analyze._on_result)
    worker.signals.failed.connect(analyze._on_failed)
    analyze._set_running(True)
    _wait_worker(pool, worker, app)
    if analyze._report is None:
        raise RuntimeError("analysis did not complete on synthetic image")

    window.tabs.setCurrentWidget(analyze)
    size = _grab_main_window(window, "gui_tab_analyze.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_analyze.png",
            "Giao diện — tab Phân tích",
            "Ảnh chụp offscreen 1280×800 sau khi chạy phân tích trên "
            "ảnh synthetic forge (mọi method bật).",
            f"QT_QPA_PLATFORM=offscreen, QWidget.grab; PNG {size} byte",
        )
    )

    blur_png = tmp_path / "blur_synthetic.png"
    blurred = cv2.GaussianBlur(
        _base_image(np.random.default_rng(SEED + 5), (360, 360)),
        (0, 0),
        sigmaX=2.6,
    )
    if not cv2.imwrite(str(blur_png), blurred):
        raise RuntimeError("could not write blurred synthetic PNG")
    enhance = window.enhance
    window.tabs.setCurrentWidget(enhance)
    enhance.load_path(blur_png)
    idx = enhance.combo_op.findData("clahe")
    if idx < 0:
        raise ValueError("CLAHE operator not found in enhance combo")
    enhance.combo_op.setCurrentIndex(idx)
    enhance._clear_result()
    enhance._run_kind = "clahe"
    enhance._run_params = dict(RESTORER_PARAMS["clahe"])
    enhance._cancel_event = threading.Event()
    enh_worker = EnhanceWorker(
        enhance._doc.array,
        "clahe",
        enhance._run_params,
        enhance._cancel_event,
    )
    enhance._worker = enh_worker
    enh_worker.signals.result.connect(enhance._on_result)
    enh_worker.signals.failed.connect(enhance._on_failed)
    enh_worker.signals.finished.connect(enhance._on_finished)
    enhance._set_running(True)
    _wait_worker(pool, enh_worker, app)
    if enhance._derived is None:
        raise RuntimeError("enhance CLAHE did not produce output")
    enhance.slider_blend.setValue(50)
    enhance._on_blend_changed(50)
    size = _grab_main_window(window, "gui_tab_enhance.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_enhance.png",
            "Giao diện — tab Tăng cường",
            "Tab xem thử với banner và blend 50% giữa ảnh blur synthetic "
            "và CLAHE (không ghi đè file gốc).",
            f"offscreen grab; PNG {size} byte",
        )
    )

    window.tabs.setCurrentIndex(2)
    window.history.refresh()
    app.processEvents()
    size = _grab_main_window(window, "gui_tab_history.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_history.png",
            "Giao diện — tab Lịch sử",
            "Một dòng lịch sử từ phiên phân tích synthetic vừa chạy (SQLite tạm).",
            f"offscreen grab; PNG {size} byte",
        )
    )

    window.tabs.setCurrentIndex(3)
    window.stats.refresh()
    app.processEvents()
    size = _grab_main_window(window, "gui_tab_stats.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_stats.png",
            "Giao diện — tab Thống kê",
            "Thống kê phiên trên cùng dữ liệu synthetic (không phải benchmark).",
            f"offscreen grab; PNG {size} byte",
        )
    )

    window.tabs.setCurrentIndex(4)
    app.processEvents()
    size = _grab_main_window(window, "gui_tab_settings.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_settings.png",
            "Giao diện — tab Cài đặt",
            "Ngưỡng đọc từ cấu hình dự án — minh họa layout báo cáo.",
            f"offscreen grab; PNG {size} byte",
        )
    )


def write_figures_md(out_dir: Path, records: list[FigureRecord]) -> None:
    lines = [
        "# Hình minh họa báo cáo (synthetic)",
        "",
        (
            "Tất cả hình trong thư mục này là **ví dụ minh họa** sinh từ ảnh "
            "**tổng hợp nội bộ** (NumPy / synthforge). Không dùng CASIA, CoMoFoD "
            "hay ảnh cá nhân. Hình **không** chứng minh độ chính xác hay kết luận "
            "ảnh bị giả mạo."
        ),
        "",
        "## Tái tạo",
        "",
        "Từ thư mục gốc repository, với virtualenv dự án và `src` trên `PYTHONPATH`:",
        "",
        "```powershell",
        "$env:QT_QPA_PLATFORM = 'offscreen'",
        "$env:PYTHONPATH = 'src'",
        "python scripts/make_report_figures.py",
        "```",
        "",
        f"Seed cố định: `{SEED}`. Mỗi PNG được nén/resize để ≤ {MAX_PNG_BYTES} byte.",
        "",
        "## Danh sách hình",
        "",
    ]
    for rec in records:
        fpath = out_dir / rec.filename
        nbytes = fpath.stat().st_size if fpath.is_file() else 0
        lines.extend(
            [
                f"### `{rec.filename}` — {rec.title_vi}",
                "",
                f"- **Nội dung:** {rec.desc_vi}",
                f"- **Cách sinh:** {rec.producer}",
                f"- **Kích thước file:** {nbytes} byte",
                "",
            ]
        )
    md_path = REPO_ROOT / "docs" / "report" / "figures.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(out_dir: Path | None = None) -> list[FigureRecord]:
    """Generate all report figures; returns metadata for logging."""
    target = _validate_out_dir(out_dir or DEFAULT_OUT)
    target.mkdir(parents=True, exist_ok=True)

    from imageguard.evaluation.synthforge import generate_set

    records: list[FigureRecord] = []
    with tempfile.TemporaryDirectory(
        prefix="ig_synth_",
        ignore_cleanup_errors=True,
    ) as tmp:
        synth_dir = Path(tmp)
        generate_set(
            n=1,
            seed=SEED,
            out_dir=synth_dir,
            size=(384, 384),
            tier="easy",
        )
        forged = next(synth_dir.glob("*_F.png"))
        if not forged.is_file():
            raise FileNotFoundError("synthforge did not emit forged PNG")

        make_pipeline_figures(target, records)
        make_gui_figures(target, forged, records)

    write_figures_md(target, records)
    return records


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    if len(sys.argv) > 2:
        raise SystemExit("usage: make_report_figures.py [out_dir]")
    out = main(dest)
    print(f"Wrote {len(out)} figures under docs/report/figures/")
