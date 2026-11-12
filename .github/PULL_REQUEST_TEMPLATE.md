## Mô tả

<!-- Một chủ đề mỗi PR. Conventional Commits trên các commit. -->

## Ranh giới khoa học

- [ ] Không thêm wording khẳng định forgery/authentic hay xác suất giả mạo.
- [ ] Không mở rộng enum `Verdict` hoặc schema báo cáo (trừ khi maintainer đã đồng ý trước).

## Phạm vi kỹ thuật

- [ ] Không sửa `configs/thresholds.yaml` / ngưỡng detector (trừ khi được giao).
- [ ] Không thêm dependency trong `pyproject.toml` (trừ khi được giao).
- [ ] Không commit `data/`, `models/`, secrets, ảnh cá nhân.

## Kiểm tra (PowerShell)

```text
QT_QPA_PLATFORM=offscreen
python -m ruff check src tests scripts
python -m pytest <đường-dẫn-test-liên-quan> -q -p no:cacheprovider
```

- Ruff:
- Pytest:

## Liên quan

<!-- Fixes #… (nếu có) -->
