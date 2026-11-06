# ImageGuard Forensics

Ứng dụng **desktop** (Python + PySide6) phân tích dấu hiệu giả mạo ảnh số — đồ án môn Xử lý ảnh.

Trọng tâm: **Copy-Move Forgery Detection** (keypoint SIFT/ORB + block-based DCT), ELA cho JPEG, noise/frequency analysis, framework đánh giá trên synthetic data + CoMoFoD.

> **Giới hạn khoa học:** Hệ thống chỉ phát hiện *dấu hiệu nghi vấn* — không kết luận ảnh đã bị giả mạo. ELA/EXIF/noise là bằng chứng khám phá, không phải bằng chứng độc lập. Khi không đủ bằng chứng, kết quả là `INCONCLUSIVE`.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
pytest
```

Yêu cầu: Python ≥ 3.13 (phát triển trên 3.14).

## Chạy app

```powershell
python -m imageguard
```

## Cấu trúc

- `src/imageguard/` — package chính (engine forensics Qt-free + UI PySide6)
- `configs/thresholds.yaml` — mọi ngưỡng thuật toán tập trung, có ghi nguồn
- `tests/` — pytest
- `docs/` — tài liệu kỹ thuật (bắt đầu từ `docs/user-guide.md`)
- `data/` — dữ liệu runtime, gồm `data/artifacts/`, **không commit**
