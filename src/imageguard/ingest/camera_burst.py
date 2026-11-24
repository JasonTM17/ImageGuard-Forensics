"""Chụp burst 4–6 frame liên tiếp từ một camera + manifest đi kèm.

Mỗi burst ghi các PNG lossless (tái dùng ``_save_lossless_png`` của
``imageguard.ingest.camera``) vào cùng thư mục capture và một manifest
``<primary-stem>.burst.json`` chỉ chứa basename — tuyệt đối không đường
dẫn tuyệt đối. ``load_burst_frames`` đọc lại an toàn: mọi tên frame phải là
basename thuần, resolve nằm gọn trong thư mục chứa manifest, từ chối ``..``,
tên tuyệt đối và số frame ngoài khoảng 4–6.

Manifest v1::

    {"version": 1, "device_index": 0, "captured_at": "<iso-utc>",
     "primary": "<primary>.png", "frames": ["<f0>.png", ...]}
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from imageguard.ingest import camera as camera_ingest

BURST_MANIFEST_VERSION = 1
BURST_MIN_FRAMES = 4
BURST_MAX_FRAMES = 6


def _validate_frame_count(frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an integer")
    if not BURST_MIN_FRAMES <= frame_count <= BURST_MAX_FRAMES:
        raise ValueError(
            f"frame_count must be in [{BURST_MIN_FRAMES}, {BURST_MAX_FRAMES}]"
        )
    return frame_count


def _is_safe_frame_name(name: object) -> bool:
    """True chỉ khi ``name`` là basename thuần — không separator/drive/``..``.

    Kiểm tra cả ``Path`` (POSIX trên Linux, Windows trên Windows) lẫn
    ``PureWindowsPath`` để bắt tên kiểu ``C:/x.png``, ``a\\b.png``, UNC
    bất kể nền tảng chạy.
    """
    if not isinstance(name, str) or not name or name != name.strip():
        return False
    if name in {".", ".."}:
        return False
    posix = Path(name)
    win = PureWindowsPath(name)
    if posix.name != name or win.name != name:
        return False
    if posix.is_absolute() or win.is_absolute() or posix.drive or win.drive:
        return False
    return ".." not in posix.parts and ".." not in win.parts


def capture_device_burst(
    device_index: int = 0,
    out_dir: Path | None = None,
    *,
    frame_count: int = 5,
    warmup_frames: int = 2,
) -> tuple[Path, Path]:
    """Chụp ``frame_count`` frame từ một camera, lưu PNG lossless + manifest.

    Trả ``(primary_path, manifest_path)``. ``primary`` là frame giữa burst —
    trùng frame mà cue temporal dùng để định vị ROI mặt — và manifest đặt
    tên ``<primary-stem>.burst.json`` ngay trong ``out_dir``. ``device_index``
    và ``warmup_frames`` được kiểm tra bên trong ``_read_frame_from_device``
    trước khi bất kỳ file nào được ghi.
    """
    count = _validate_frame_count(frame_count)
    target_dir = (
        Path(out_dir)
        if out_dir is not None
        else camera_ingest.DEFAULT_CAPTURE_DIR
    )

    saved: list[Path] = []
    try:
        for i in range(count):
            frame = camera_ingest._read_frame_from_device(
                device_index, warmup_frames
            )
            prefix = f"camera_burst_dev{int(device_index)}_f{i:02d}"
            saved.append(
                camera_ingest._save_lossless_png(frame, target_dir, prefix)
            )
    except Exception:
        # Burst dở dang không có manifest → các PNG đã ghi là mồ côi, dọn lại.
        for partial in saved:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    primary = saved[count // 2]
    manifest = {
        "version": BURST_MANIFEST_VERSION,
        "device_index": int(device_index),
        "captured_at": datetime.now(UTC).isoformat(),
        "primary": primary.name,
        "frames": [p.name for p in saved],
    }
    manifest_path = target_dir / f"{primary.stem}.burst.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, default=str, indent=1) + "\n",
        encoding="utf-8",
    )
    return primary, manifest_path


def load_burst_frames(manifest_path: Path) -> list[np.ndarray]:
    """Đọc manifest burst và trả danh sách frame BGR theo đúng thứ tự ghi.

    Chỉ resolve tên file bên trong thư mục chứa manifest; từ chối ``..``,
    tên tuyệt đối, manifest sai schema/version và số frame ngoài 4–6.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"burst manifest không tồn tại: {mpath.name}")
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"burst manifest không đọc được: {e}") from e
    if not isinstance(data, dict):
        raise TypeError("burst manifest phải là JSON object")
    if data.get("version") != BURST_MANIFEST_VERSION:
        raise ValueError(
            f"burst manifest version không hỗ trợ: {data.get('version')!r}"
        )

    names = data.get("frames")
    if not isinstance(names, list) or not all(
        isinstance(n, str) for n in names
    ):
        raise ValueError("burst manifest 'frames' phải là list[str]")
    if not BURST_MIN_FRAMES <= len(names) <= BURST_MAX_FRAMES:
        raise ValueError(
            f"burst manifest phải có {BURST_MIN_FRAMES}–{BURST_MAX_FRAMES} "
            f"frames, thực tế {len(names)}"
        )
    primary = data.get("primary")
    if primary is not None and not _is_safe_frame_name(primary):
        raise ValueError("burst manifest 'primary' không phải basename hợp lệ")
    if not all(_is_safe_frame_name(n) for n in names):
        raise ValueError("burst manifest chứa frame name không an toàn")

    base = mpath.parent
    resolved_base = base.resolve()
    frames: list[np.ndarray] = []
    for name in names:
        candidate = base / name
        # Phòng symlink/frame name thoát khỏi thư mục manifest.
        if candidate.resolve().parent != resolved_base:
            raise ValueError(
                f"frame burst thoát khỏi thư mục manifest: {name}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"thiếu frame burst: {name}")
        buf = np.frombuffer(candidate.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"không decode được frame burst: {name}")
        frames.append(img)
    return frames
