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

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "report" / "figures"
SEED = 20261112
GUI_SIZE = (1280, 800)
MAX_PNG_BYTES = 300 * 1024
GUI_FONT_CANDIDATES = ("Segoe UI", "Tahoma", "Arial")
FONT_PROBE_VI = "Ảnh chụp báo cáo — Phân tích"
_LAST_GUI_PLATFORM = ""

# Recorded for figures.md after a successful GUI capture run.
_LAST_GUI_FONT = ""


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


def _resize_to_fit(source: np.ndarray, scale: float) -> np.ndarray:
    """Downscale ``source`` by ``scale``; always loses at least 1px per axis."""
    h, w = source.shape[:2]
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if nw == w and nh == h:
        nw, nh = max(1, w - 1), max(1, h - 1)
    return cv2.resize(source, (nw, nh), interpolation=cv2.INTER_AREA)


def write_png_limited(bgr: np.ndarray, path: Path, max_bytes: int = MAX_PNG_BYTES) -> int:
    """Encode BGR to PNG, downscaling until the payload is at most ``max_bytes``.

    Returns the number of bytes written. Raises ``ValueError`` (leaving the
    previous file untouched) when no size fits, so an oversized PNG never
    reaches disk.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("image must be H×W×3 BGR")
    if bgr.size == 0:
        raise ValueError("image must be non-empty")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, np.integer)):
        raise TypeError("max_bytes must be an integer")
    cap = int(max_bytes)
    if cap <= 0:
        raise ValueError("max_bytes must be positive")

    source = np.ascontiguousarray(bgr)
    scale = 1.0
    img = source
    while True:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError("OpenCV could not encode PNG")
        payload = buf.tobytes()
        if len(payload) <= cap:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return len(payload)
        if min(img.shape[:2]) <= 1:
            raise ValueError(
                f"cannot encode PNG within {cap} bytes; "
                f"smallest attempt was {len(payload)} bytes at "
                f"{img.shape[1]}×{img.shape[0]}"
            )
        scale *= 0.85
        img = _resize_to_fit(source, scale)


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


def _gui_platform_label() -> str:
    """Human-readable Qt platform name for docs (no host paths)."""
    explicit = os.environ.get("QT_QPA_PLATFORM")
    if explicit:
        return explicit
    if sys.platform == "win32":
        return "windows (default)"
    return "native default"


def _prepare_gui_platform() -> str:
    """Ensure GUI grabs use a font-capable Qt platform (not offscreen)."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ.pop("QT_QPA_PLATFORM", None)
    label = _gui_platform_label()
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        raise RuntimeError(
            "GUI report figures need a real Qt platform with system fonts. "
            "Unset QT_QPA_PLATFORM=offscreen before running this script."
        )
    return label


def _apply_gui_font(app: object) -> str:
    """Pick an installed font with Vietnamese/Latin glyphs and verify a text grab."""
    from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
    from PySide6.QtWidgets import QLabel

    families = set(QFontDatabase.families())
    chosen = ""
    for family in GUI_FONT_CANDIDATES:
        if family not in families:
            continue
        font = QFont(family, 10)
        metrics = QFontMetrics(font)
        if metrics.horizontalAdvance(FONT_PROBE_VI) < 80:
            continue
        app.setFont(font)
        chosen = family
        break
    if not chosen:
        raise RuntimeError(
            "Cannot render Vietnamese UI text: none of "
            f"{GUI_FONT_CANDIDATES} produced glyph metrics on this system."
        )

    label = QLabel(FONT_PROBE_VI)
    label.setFont(QFont(chosen, 11))
    label.adjustSize()
    app.processEvents()
    gray = cv2.cvtColor(_qpixmap_to_bgr(label.grab()), cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 12.0:
        raise RuntimeError(
            "Font probe render is blank (typical of offscreen Qt without fonts). "
            "Run on native Windows Qt without QT_QPA_PLATFORM=offscreen."
        )
    return chosen


def _verify_gui_screenshot_png(path: Path) -> None:
    """Reject grabs that lack ink in tab bar / disclaimer (tofu or empty UI)."""
    data = path.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not decode GUI screenshot {path.name}")
    height = bgr.shape[0]
    bands = (
        ("tab bar", bgr[0 : min(48, height), :, :]),
        ("footer", bgr[max(0, height - 44) :, :, :]),
    )
    for region_name, band in bands:
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        dark_fraction = float((gray < 160).mean())
        if dark_fraction < 0.008:
            raise RuntimeError(
                f"{path.name}: {region_name} has no readable text "
                "(missing fonts or offscreen tofu); not writing usable report figures."
            )


def _seed_history_rows(
    analysis: object,
    tmp_path: Path,
    *,
    extra_count: int = 2,
) -> None:
    """Persist extra synthetic analyses so History/Stats tabs are populated."""
    from imageguard.core.models import ForensicReport
    from imageguard.evaluation.synthforge import generate_set
    from imageguard.services.analysis_service import AnalysisRequest

    if extra_count < 1:
        raise ValueError("extra_count must be >= 1")
    hist_dir = tmp_path / "history_synth"
    generate_set(
        n=extra_count,
        seed=SEED + 101,
        out_dir=hist_dir,
        size=(352, 352),
        tier="easy",
    )
    forged_paths = sorted(hist_dir.glob("*_F.png"))
    if len(forged_paths) < extra_count:
        raise FileNotFoundError("synthforge did not emit enough forged PNGs for history")
    methods = ["keypoint", "block", "ela"]
    for forged in forged_paths[:extra_count]:
        report = analysis.analyze(AnalysisRequest(path=forged, methods=methods))
        if not isinstance(report, ForensicReport):
            detail = getattr(report, "message", report)
            raise ValueError(f"history seed analysis failed: {detail}")  # noqa: TRY004


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
    out_path = out_dir / filename
    bgr = _qpixmap_to_bgr(window.grab())
    size = write_png_limited(bgr, out_path)
    _verify_gui_screenshot_png(out_path)
    return size


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
    global _LAST_GUI_PLATFORM, _LAST_GUI_FONT

    platform_label = _prepare_gui_platform()
    from PySide6.QtWidgets import QApplication

    from imageguard.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    _LAST_GUI_FONT = _apply_gui_font(app)
    _LAST_GUI_PLATFORM = platform_label
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

    _seed_history_rows(analysis, tmp_path, extra_count=2)

    window.tabs.setCurrentWidget(analyze)
    gui_note = (
        f"Qt {_LAST_GUI_PLATFORM}, QFont({_LAST_GUI_FONT}), "
        f"MainWindow {GUI_SIZE[0]}×{GUI_SIZE[1]} grab"
    )
    size = _grab_main_window(window, "gui_tab_analyze.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_analyze.png",
            "Giao diện — tab Phân tích",
            "Ảnh chụp native 1280×800 sau khi chạy phân tích trên "
            "ảnh synthetic forge (mọi method bật).",
            f"{gui_note}; PNG {size} byte",
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
            f"{gui_note}; PNG {size} byte",
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
            "Ba dòng lịch sử từ phiên phân tích synthetic (AnalysisService, SQLite tạm).",
            f"{gui_note}; PNG {size} byte",
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
            f"{gui_note}; PNG {size} byte",
        )
    )

    window.tabs.setCurrentIndex(4)
    app.processEvents()
    size = _grab_main_window(window, "gui_tab_settings.png", out_dir)
    records.append(
        FigureRecord(
            "gui_tab_settings.png",
            "Giao diện — tab Cài đặt",
            "Cây ngưỡng đọc từ cấu hình dự án — minh họa layout báo cáo.",
            f"{gui_note}; PNG {size} byte",
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
        "$env:PYTHONPATH = 'src'",
        "# Không đặt QT_QPA_PLATFORM=offscreen — tab GUI cần nền tảng Windows có font.",
        "python scripts/make_report_figures.py",
        "```",
        "",
        (
            "Tab GUI: nền tảng Qt "
            f"`{_LAST_GUI_PLATFORM or _gui_platform_label()}`"
            + (
                f", font `{_LAST_GUI_FONT}`."
                if _LAST_GUI_FONT
                else " với font hệ thống (ưu tiên Segoe UI)."
            )
        ),
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
