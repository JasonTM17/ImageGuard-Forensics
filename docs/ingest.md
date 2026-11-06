# Folder ingest

Folder ingest watches one directory for finished still images and sends them
through `AnalysisService` serially. It ignores hidden names and files ending in
`.tmp`, `.partial`, or `.filepart`, checks that file size and modification time
stay stable across two observations, and sniffs image magic before analysis.
It polls with the Python standard library and leaves input files in place.

Use `python -m imageguard ingest once <dir>` for a single scan or
`python -m imageguard ingest watch <dir>` to keep polling. Add
`--config <file>` after the directory to select a config. The example is
[`docs/examples/ingest.example.yaml`](examples/ingest.example.yaml). If no
config is supplied, `data/ingest/config.yaml` is loaded when present; otherwise
built-in defaults are used. The real config and the default custody log live
under the gitignored `data/` directory.

The config supports `camera_label`, `max_pending` (default `1`), `methods`
(default `keypoint`, `block`, `ela`), a cwd-relative `data_dir` (default
`data`), an optional `consent_note`, and an optional raw `http` mapping. When
enabled, an available later HTTP puller may place finished JPEGs in the watched
folder; folder ingest itself does not fetch network content.

Each JSONL line in `data/ingest/custody.jsonl` records `record_version`,
`event_id`, UTC `event_time`, `event`, `image_sha256`, basename-only `basename`,
`camera_label` with its `operator_supplied` trust label, `file_mtime` with its
`filesystem_unverified` trust label, `analysis_id`, the threshold `config_hash`,
and `tool_version`. A configured `consent_note` is copied as operator text, not
as legal proof. Events are appended; earlier lines are not rewritten.

The custody log is not tamper-proof. The camera label is supplied by the
operator, and the file modification time is filesystem-unverified. The log
does not establish an authenticated camera or a verified capture time.
