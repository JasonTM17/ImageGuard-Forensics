# Database — SQLite + artifact store

## Quyết định

- SQLite tại `data/imageguard.db` (gitignored) — single-user desktop, không
  cần server DB.
- SQLAlchemy 2.x ORM, WAL mode, `PRAGMA foreign_keys=ON`.
- **Không Alembic**: schema nhỏ; thay đổi = bump `SCHEMA_VERSION` + recreate.
  `make_session_factory` raise lỗi rõ nếu version lệch.
- DB chỉ lưu **relative paths** — `_assert_relative` chặn absolute/`..` khi ghi.

## ERD

```
image_assets 1───n analyses 1───n method_results
                        └───n artifacts
experiments 1───n experiment_results
meta(key,value): schema_version
```

| Bảng | Vai trò |
|---|---|
| image_assets | dedup theo sha256 — cùng file = cùng row |
| analyses | một lần phân tích: verdict, config_hash, duration |
| method_results | per-method: verdict/score/score_definition/evidence/limitations |
| artifacts | rel_path tới `data/artifacts/<analysis_id>/` |
| experiments + experiment_results | eval runs + metrics JSON |

## Artifact files

`data/artifacts/<analysis_id>/{method}_mask.png`, `{method}_heat.png`,
`report.json`. Delete analysis → cascade rows + `ArtifactStore.delete_analysis`
dọn thư mục file.
