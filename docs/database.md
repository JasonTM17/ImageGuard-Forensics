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
image_assets 1───n analyses
                   ├───n method_results
                   └───n artifacts
experiments 1───n experiment_results
meta(key,value): schema_version
```

| Bảng | Vai trò |
|---|---|
| image_assets | dedup theo sha256 — cùng file = cùng row |
| analyses | một lần phân tích: `overall_verdict`, `config_hash`, `duration_ms` |
| method_results | per-method: `verdict`/`score`/`score_definition`/`evidence_json`/`limitations_json` |
| artifacts | rel_path tới `data/artifacts/<analysis_id>/`; kind `mask`/`heatmap`/`pairs`/`report` (schema khai báo thêm `ela`) |
| experiments + experiment_results | eval runs + `metrics_json` |
| meta | key/value — `schema_version` |

## Artifact files

`data/artifacts/<analysis_id>/{method}_mask.png`, `{method}_heat.png`,
`{method}_pairs.png` (khi method có `pairs_image`), `report.json`.
`export_markdown` ghi thêm `report.md` vào cùng thư mục. Delete analysis →
cascade rows + `ArtifactStore.delete_analysis` dọn thư mục file.
