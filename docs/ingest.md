# Folder ingest

Folder ingest watches one directory for finished still images and sends them
through `AnalysisService` serially. It ignores hidden names and files ending in
`.tmp`, `.partial`, or `.filepart`, checks that file size and modification time
stay stable across two observations, and sniffs image magic before analysis.
Symlinks and Windows reparse points (junctions) in the watched folder are not
followed: each link is logged once as an `error` with `PATH_INVALID` and left
on disk. Before SHA-256 hashing, file size is compared to the shared
`MAX_FILE_BYTES` limit from `configs/thresholds.yaml` (`limits`); oversize files
are logged once as `FILE_TOO_LARGE` without reading the full byte stream. After
analysis, the report `image_sha256` must match the scan-time hash; a mismatch
is recorded as `FILE_UNREADABLE` instead of `analyzed`. It polls with the
Python standard library and leaves input files in place.

Use `python -m imageguard ingest once <dir>` for a single scan or
`python -m imageguard ingest watch <dir>` to keep polling. Add
`--config <file>` after the directory to select a config. The example is
[`docs/examples/ingest.example.yaml`](examples/ingest.example.yaml). If no
config is supplied, `data/ingest/config.yaml` is loaded when present; otherwise
built-in defaults are used. The real config and the default custody log live
under the gitignored `data/` directory.

The config supports `camera_label`, `max_pending` (default `1`), `methods`
(default `keypoint`, `block`, `ela`), a cwd-relative `data_dir` (default
`data`), an optional `consent_note`, and an optional raw `http` mapping.
`data_dir` must be relative to the process working directory when the config is
loaded: absolute paths, `..` segments, drive-relative paths (for example
`D:folder`), and root-relative Windows paths (for example `\private`) are
rejected, and the resolved `data_dir` must remain under the resolved working
directory. When enabled, the HTTP snapshot puller may place finished JPEGs in
the watched folder. Folder scanning itself does not open network connections.

## Summary line and deferred files

Every scan prints one summary line in exactly this shape:

```text
ingest once: analyzed=2 errors=0 skipped_duplicate=1 deferred=3
```

`ingest watch` prints the same line with `watch` in place of `once`, and only
for scans that had activity. `analyzed` counts files submitted to
`AnalysisService` and reported without error, `errors` counts submitted files
that came back as an error, and `skipped_duplicate` counts files whose SHA-256
was already seen in the same scan.

`max_pending` in the ingest YAML (top-level key, positive integer, default `1`)
caps how many files **one scan** submits for analysis. Every remaining
candidate is **deferred**: it is recorded as a `deferred` custody event and
left on disk, but it has **not** been analyzed, and its bytes have **not** been
decoded or validated — only the magic-byte sniff and the file-stability check
ran. Therefore `errors=0` together with `deferred>0` does **not** mean every
file in the folder is valid: a corrupt image among the deferred files only
produces an `error` (and increments `errors`) once a later scan actually
submits it.

`deferred` counts files newly marked deferred by that scan. A file that was
already marked `deferred` in an earlier scan is not counted again even while it
still waits, so `deferred=0` alone is not proof that nothing is left.

To process the rest, rerun `ingest once <dir>` (files with a prior `deferred`
event are ordered first in the next scan), or raise `max_pending` in the ingest
YAML so one scan submits more files. `ingest watch` repeats the same per-scan
budget on each poll.

Optional boolean `enhance_sibling` (default `false`): when `true`, after a
successful ingest analysis the scanner loads the same source file with
`safe_open`, runs the classical unsharp derive used elsewhere, and writes
`<stem>.view.png` plus `<stem>.view.json` next to it (never renaming or
overwriting the original). Those `.view.*` names are ignored on later scans so
they are not analyzed again; `AnalysisService` still receives only the original
path. Failures are noted on the custody line as the stable tag
`enhance_sibling_failed` (no exception text and no path) without aborting
the run. The sibling is a viewing file. It is not parent evidence, and ingest
does not analyze it.

Each JSONL line in `data/ingest/custody.jsonl` records `record_version`,
`event_id`, UTC `event_time`, `event`, `image_sha256`, basename-only `basename`,
`camera_label` with its `operator_supplied` trust label, `file_mtime` with its
`filesystem_unverified` trust label, `analysis_id`, the threshold `config_hash`,
`tool_version`, and an optional `error_code` (error events and
`enhance_sibling_failed`). A configured `consent_note` is copied as operator
text, not as legal proof. Events are appended; earlier lines are not rewritten.

The custody log is not tamper-proof. The camera label is supplied by the
operator, and the file modification time is filesystem-unverified. The log
does not establish an authenticated camera or a verified capture time.
