"""Thu nhận khung hình thô từ Webcam máy tính, Camera USB ngoài và Camera IP/IoT.

Mô-đun Qt-free giúp lấy trực tiếp frame cảm biến:
- Webcam nội bộ (``device_index=0``) hoặc Camera USB ngoài (``device_index >= 1``)
  thông qua ``cv2.VideoCapture`` (ưu tiên ``CAP_DSHOW`` trên Windows để mở
  nhanh và ổn định nhiều camera cùng lúc), lưu dạng PNG không nén mất dữ liệu
  (lossless) để tránh tái nén JPEG làm nhiễu dấu vết Forensics/Deepfake.
- Hỗ trợ chụp song song cả 2 camera (``capture_dual_device_frames``) ghép
  cạnh nhau thành 1 ảnh đối chiếu.
- Camera IP / IoT qua HTTP/HTTPS JPEG Snapshot bằng ``pull_due`` trong
  ``http_pull.py`` (giữ đầy đủ kiểm tra SSRF, private metadata block, magic bytes).
"""

from __future__ import annotations

import secrets
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from imageguard.core.paths import data_root
from imageguard.ingest.http_pull import _normalize_host, pull_due

DEFAULT_CAPTURE_DIR = data_root() / "captures"


def _validate_non_negative_int(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _open_capture(device_index: int):
    """Mở VideoCapture; ưu tiên CAP_DSHOW trên Windows và fallback mượt cho mock/OS khác."""
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        try:
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        except TypeError:
            pass
    return cv2.VideoCapture(device_index)


def _read_frame_from_device(device_index: int, warmup_frames: int) -> np.ndarray:
    """Mở thiết bị, đọc ``warmup_frames`` để ổn định phơi sáng rồi trả về frame BGR uint8."""
    idx = _validate_non_negative_int(device_index, "device_index", minimum=0)
    warmup = _validate_non_negative_int(warmup_frames, "warmup_frames", minimum=1)

    cap = _open_capture(idx)
    frame: np.ndarray | None = None
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Không mở được thiết bị camera (index={idx}). "
                "Kiểm tra kết nối webcam/camera ngoài."
            )
        for _ in range(warmup):
            ok, candidate = cap.read()
            if (
                ok
                and isinstance(candidate, np.ndarray)
                and candidate.size > 0
                and candidate.ndim == 3
                and candidate.shape[2] == 3
                and candidate.dtype == np.uint8
            ):
                frame = candidate
        if frame is None:
            raise RuntimeError(
                f"Không đọc được khung hình hợp lệ từ camera (index={idx})."
            )
        return frame
    finally:
        cap.release()


def _save_lossless_png(frame: np.ndarray, out_dir: Path, prefix: str) -> Path:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    ok_enc, encoded = cv2.imencode(".png", frame)
    if not ok_enc:
        raise RuntimeError("Không mã hóa được khung hình camera sang PNG.")

    utc_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    token8 = secrets.token_hex(4)
    out_path = target_dir / f"{prefix}_{utc_stamp}_{token8}.png"
    out_path.write_bytes(encoded.tobytes())
    return out_path


def capture_device_frame(
    device_index: int = 0,
    out_dir: Path | None = None,
    *,
    warmup_frames: int = 2,
) -> Path:
    """Chụp một khung hình thô từ webcam/camera USB theo ``device_index`` và lưu PNG."""
    target_dir = Path(out_dir) if out_dir is not None else data_root() / "captures"
    frame = _read_frame_from_device(device_index, warmup_frames)
    return _save_lossless_png(frame, target_dir, f"camera_dev{int(device_index)}")


def capture_dual_device_frames(
    primary_index: int = 0,
    secondary_index: int = 1,
    out_dir: Path | None = None,
    *,
    warmup_frames: int = 3,
) -> Path:
    """Chụp song song 2 camera (Webcam nội bộ + Camera ngoài) và ghép cạnh nhau thành PNG."""
    idx_a = _validate_non_negative_int(primary_index, "primary_index", minimum=0)
    idx_b = _validate_non_negative_int(secondary_index, "secondary_index", minimum=0)
    if idx_a == idx_b:
        raise ValueError("primary_index và secondary_index phải khác nhau.")

    target_dir = Path(out_dir) if out_dir is not None else data_root() / "captures"
    frame_a = _read_frame_from_device(idx_a, warmup_frames)
    frame_b = _read_frame_from_device(idx_b, warmup_frames)
    if frame_b.shape[0] != frame_a.shape[0]:
        target_h = frame_a.shape[0]
        target_w = max(1, round(frame_b.shape[1] * target_h / max(1, frame_b.shape[0])))
        interp = cv2.INTER_AREA if target_h < frame_b.shape[0] else cv2.INTER_CUBIC
        frame_b = cv2.resize(frame_b, (target_w, target_h), interpolation=interp)

    combined = np.hstack([frame_a, frame_b])
    return _save_lossless_png(combined, target_dir, f"camera_dual_dev{idx_a}_dev{idx_b}")


def capture_iot_http_frame(
    url: str,
    out_dir: Path | None = None,
    *,
    timeout: float = 5.0,
    allow_loopback: bool = False,
    allow_link_local: bool = False,
) -> Path:
    """Lấy 1 snapshot JPEG từ Camera IP / IoT qua ``http_pull.pull_due``."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL camera không được để trống.")
    cleaned_url = url.strip()
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < float(timeout) <= 60.0
    ):
        raise ValueError("timeout phải nằm trong (0, 60] giây.")

    parsed = urllib.parse.urlsplit(cleaned_url)
    host = _normalize_host(parsed.hostname or "")
    if host is None:
        raise ValueError("URL camera không hợp lệ (thiếu hostname/IP).")

    target_dir = Path(out_dir) if out_dir is not None else data_root() / "captures"
    config = {
        "enabled": True,
        "urls": [cleaned_url],
        "hosts": [host],
        "interval_s": 5.0,
        "timeout_s": float(timeout),
        "max_retries": 0,
        "allow_loopback": bool(allow_loopback),
        "allow_link_local": bool(allow_link_local),
    }
    results = pull_due(config, target_dir, now=datetime.now(UTC))
    if not results:
        raise RuntimeError("Không nhận được phản hồi từ module HTTP camera.")
    res = results[0]
    if res.get("status") != "written":
        reason = res.get("reason", "unknown")
        raise RuntimeError(f"Không lấy được snapshot từ Camera IoT: {reason}")
    return target_dir / res["basename"]


def probe_camera_indices(max_index: int = 3) -> list[int]:
    """Dò nhanh danh sách index camera (0..max_index-1) đang mở được."""
    limit = max(0, int(max_index))
    available: list[int] = []
    for idx in range(limit):
        cap = _open_capture(idx)
        try:
            if cap.isOpened():
                available.append(idx)
        finally:
            cap.release()
    return available
