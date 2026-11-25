"""Chụp burst 4–6 frame cách nhau trong vài giây từ một camera + manifest.

Mỗi burst ghi các PNG lossless (tái dùng ``_save_lossless_png`` của
``imageguard.ingest.camera``) vào cùng thư mục capture và một manifest
``<primary-stem>.burst.json`` chỉ chứa basename — tuyệt đối không đường
dẫn tuyệt đối. ``load_burst_frames`` đọc lại an toàn: mọi tên frame phải là
basename thuần, resolve nằm gọn trong thư mục chứa manifest, từ chối ``..``,
tên tuyệt đối và số frame ngoài khoảng 4–6.

Manifest v2::

    {"version": 2, "device_index": 0, "captured_at": "<iso-utc>",
     "primary": "<primary>.png", "primary_sha256": "<hex>",
     "frame_interval_s": 0.75, "frames": ["<f0>.png", ...],
     "sha256": {"<f0>.png": "<hex>", ...}}
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from imageguard.core.errors import ErrorResult
from imageguard.core.io import safe_open
from imageguard.ingest import camera as camera_ingest

BURST_MANIFEST_VERSION = 2
BURST_MIN_FRAMES = 4
BURST_MAX_FRAMES = 6
BURST_FRAME_INTERVAL_S = 0.75
BURST_MANIFEST_MAX_BYTES = 64 * 1024
BURST_WORK_MAX_SIDE = 640
BURST_MAX_TOTAL_WORK_PIXELS = BURST_MAX_FRAMES * BURST_WORK_MAX_SIDE * BURST_WORK_MAX_SIDE


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
    frame_interval_s: float = BURST_FRAME_INTERVAL_S,
) -> tuple[Path, Path]:
    """Chụp ``frame_count`` frame cách nhau từ một camera, lưu PNG lossless + manifest.

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
        frames, read_times = camera_ingest._read_spaced_frame_burst_from_device(
            device_index, warmup_frames, count, frame_interval_s
        )
        for i, frame in enumerate(frames):
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
    manifest_path = target_dir / f"{primary.stem}.burst.json"
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        frame_sha256 = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in saved
        }
        first_read = read_times[0] if read_times else 0.0
        manifest = {
            "version": BURST_MANIFEST_VERSION,
            "device_index": int(device_index),
            "captured_at": datetime.now(UTC).isoformat(),
            "primary": primary.name,
            "primary_sha256": frame_sha256[primary.name],
            "frame_interval_s": round(float(frame_interval_s), 3),
            "frame_read_offsets_s": [round(t - first_read, 3) for t in read_times],
            "frames": [p.name for p in saved],
            "sha256": frame_sha256,
        }
        tmp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, default=str, indent=1) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_manifest, manifest_path)
    except Exception:
        tmp_manifest.unlink(missing_ok=True)
        for partial in saved:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return primary, manifest_path


def load_burst_frames(
    manifest_path: Path, *, expected_primary_sha256: str | None = None
) -> list[np.ndarray]:
    """Đọc manifest burst và trả danh sách frame BGR theo đúng thứ tự ghi.

    Chỉ resolve tên file bên trong thư mục chứa manifest; từ chối ``..``,
    tên tuyệt đối, symlink, manifest sai schema/version và frame vượt giới hạn
    byte/pixel của ``safe_open``. Khi caller truyền ``expected_primary_sha256``,
    manifest phải chứng minh nó thuộc đúng primary PNG đang phân tích.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        raise FileNotFoundError(f"burst manifest không tồn tại: {mpath.name}")
    try:
        with mpath.open("rb") as f:
            if os.fstat(f.fileno()).st_size > BURST_MANIFEST_MAX_BYTES:
                raise ValueError("burst manifest quá lớn")
            raw = f.read(BURST_MANIFEST_MAX_BYTES + 1)
    except OSError as e:
        raise ValueError(f"burst manifest không đọc được: {e}") from e
    if len(raw) > BURST_MANIFEST_MAX_BYTES:
        raise ValueError("burst manifest quá lớn")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
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
    if not _is_safe_frame_name(primary) or primary not in names:
        raise ValueError("burst manifest 'primary' không hợp lệ")
    if primary != names[len(names) // 2]:
        raise ValueError("burst manifest 'primary' phải là frame giữa")
    if len(set(names)) != len(names):
        raise ValueError("burst manifest chứa frame name trùng lặp")
    if not all(_is_safe_frame_name(n) for n in names):
        raise ValueError("burst manifest chứa frame name không an toàn")
    interval = data.get("frame_interval_s")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not 0.0 < float(interval) <= 5.0
    ):
        raise ValueError("burst manifest 'frame_interval_s' không hợp lệ")
    offsets = data.get("frame_read_offsets_s")
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(names)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) < 0.0
            for v in offsets
        )
        or any(float(offsets[i]) > float(offsets[i + 1]) for i in range(len(offsets) - 1))
    ):
        raise ValueError("burst manifest 'frame_read_offsets_s' không hợp lệ")

    sha_map = data.get("sha256")
    if sha_map is None:
        raise ValueError("burst manifest thiếu sha256 map")
    if (
        not isinstance(sha_map, dict)
        or any(
            not isinstance(k, str) or k not in names or
            not isinstance(v, str) or len(v) != 64 or
            any(c not in "0123456789abcdefABCDEF" for c in v)
            for k, v in sha_map.items()
        )
        or any(name not in sha_map for name in names)
    ):
        raise ValueError("burst manifest 'sha256' không hợp lệ")
    primary_sha256 = data.get("primary_sha256")
    if expected_primary_sha256 is not None and (
        not isinstance(primary_sha256, str)
        or primary_sha256.lower() != expected_primary_sha256.lower()
        or sha_map.get(primary, "").lower() != expected_primary_sha256.lower()
    ):
        raise ValueError("burst manifest không khớp primary PNG đang phân tích")

    base = mpath.parent
    resolved_base = base.resolve()
    frames: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    total_work_pixels = 0
    for name in names:
        candidate = base / name
        # Phòng symlink/frame name thoát khỏi thư mục manifest.
        if candidate.resolve().parent != resolved_base:
            raise ValueError(
                f"frame burst thoát khỏi thư mục manifest: {name}"
            )
        if candidate.is_symlink():
            raise ValueError(f"frame burst symlink bị từ chối: {name}")
        if not candidate.is_file():
            raise FileNotFoundError(f"thiếu frame burst: {name}")
        doc = safe_open(candidate)
        if isinstance(doc, ErrorResult):
            raise ValueError(  # noqa: TRY004 - ErrorResult mã hoá nội dung file không an toàn, không phải sai kiểu Python
                f"frame burst không vượt kiểm tra an toàn: {name} ({doc.code.value})"
            )
        if doc.sha256.lower() != sha_map[name].lower():
            raise ValueError(f"frame burst không khớp sha256 manifest: {name}")
        arr = doc.array
        shape = arr.shape[:2]
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("frame burst không đồng nhất kích thước")
        h, w = shape
        scale = min(1.0, BURST_WORK_MAX_SIDE / float(max(h, w)))
        if scale < 1.0:
            arr = cv2.resize(
                arr,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        total_work_pixels += int(arr.shape[0]) * int(arr.shape[1])
        if total_work_pixels > BURST_MAX_TOTAL_WORK_PIXELS:
            raise ValueError("burst vượt ngân sách pixel tổng")
        frames.append(arr)
    return frames
