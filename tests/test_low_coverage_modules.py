"""Targeted edge and branch tests for lower-coverage modules:
nafnet_arm, blind_kernel, modelforge, enhance_study, derive,
ingest.folder, ingest.http_pull, and cli."""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from imageguard import cli
from imageguard.core.config import load_thresholds
from imageguard.core.errors import ErrorCode, ErrorResult
from imageguard.enhancement.derive import _json_safe, write_sibling
from imageguard.evaluation import blind_kernel, enhance_study, modelforge, nafnet_arm
from imageguard.evaluation.enhance_ops import disk_kernel, motion_kernel
from imageguard.ingest.config import IngestConfig
from imageguard.ingest.folder import FolderIngestor
from imageguard.ingest.http_pull import pull_due
from imageguard.services.analysis_service import AnalysisService


def test_nafnet_arm_validation_preflight_budget_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(TypeError, match="n must"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=True)
    with pytest.raises(ValueError, match="positive"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=0)
    with pytest.raises(ValueError, match="tier"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=1, tier="unknown")
    with pytest.raises(TypeError, match="size must"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=1, size=True)
    with pytest.raises(ValueError, match="positive"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=1, size=0)
    with pytest.raises(TypeError, match="seed must"):
        nafnet_arm.run_nafnet_arm(tmp_path, n=1, seed=False)

    # 1. Missing weights preflight -> NOT_RUN
    monkeypatch.setattr(nafnet_arm, "_preflight", lambda: (0.05, None, FileNotFoundError("no.onnx")))
    out_missing = tmp_path / "missing"
    res_missing = nafnet_arm.run_nafnet_arm(out_missing, n=2)
    assert res_missing["status"] == "NOT_RUN"
    assert "NAFNet weight is missing" in res_missing["reason"]
    assert (out_missing / "nafnet_arm.json").is_file()
    assert "# NAFNet E3 Viewing Arm" in (out_missing / "nafnet_arm.md").read_text(encoding="utf-8")

    # 2. Zero-sized spatial feature preflight -> NOT_RUN
    monkeypatch.setattr(
        nafnet_arm,
        "_preflight",
        lambda: (0.05, None, RuntimeError("Cannot use 'edge' mode to pad dimension with a value of 0")),
    )
    res_zero = nafnet_arm.run_nafnet_arm(tmp_path / "zero_spatial", n=1)
    assert res_zero["status"] == "NOT_RUN"
    assert "zero-sized spatial features" in res_zero["reason"]

    # 3. Preflight exceeds time cap -> NOT_RUN
    monkeypatch.setattr(
        nafnet_arm,
        "_preflight",
        lambda: (nafnet_arm.MAX_PREFLIGHT_SECONDS + 1.0, {"name": "NAFNet"}, None),
    )
    res_slow = nafnet_arm.run_nafnet_arm(tmp_path / "slow", n=1)
    assert res_slow["status"] == "NOT_RUN"
    assert "30-second runtime limit" in res_slow["reason"]

    # 4. Budget sample-count reduction + DONE_WITH_CONCERNS + CLI main()
    monkeypatch.setattr(
        nafnet_arm,
        "_preflight",
        lambda: (25.0, {"name": "NAFNet", "model_bytes": 128}, None),
    )
    monkeypatch.setattr(
        nafnet_arm,
        "derive",
        lambda img, kind, _params: (img.copy(), {"kind": kind}),
    )

    def _fake_study(set_dir, eval_dir, **kwargs):
        return {
            "n": 2,
            "seed": kwargs["seed"],
            "degradations": ["blur"],
            "restorers": kwargs["restorers"],
            "methods": list(kwargs["methods"]),
            "samples": [],
            "aggregates": {},
        }

    monkeypatch.setattr(enhance_study, "run_study", _fake_study)
    monkeypatch.setattr(enhance_study, "_render_markdown", lambda _p: "# Classical Enhancement Study\n")
    res_budget = nafnet_arm.run_nafnet_arm(tmp_path / "budget", n=10, size=64)
    assert res_budget["status"] == "DONE_WITH_CONCERNS"
    assert "runtime_note" in res_budget
    assert res_budget["n_requested"] == 10

    # 5. Study exception -> NOT_RUN
    def _boom_study(*_args, **_kwargs):
        raise RuntimeError("study boom")

    monkeypatch.setattr(enhance_study, "run_study", _boom_study)
    res_err = nafnet_arm.run_nafnet_arm(tmp_path / "err", n=1, size=64)
    assert res_err["status"] == "NOT_RUN"
    assert "RuntimeError" in res_err["reason"]

    # 6. CLI main()
    monkeypatch.setattr(enhance_study, "run_study", _fake_study)
    monkeypatch.setattr(
        nafnet_arm,
        "_preflight",
        lambda: (0.1, {"name": "NAFNet", "model_bytes": 128}, None),
    )
    cli_out = tmp_path / "cli_arm"
    nafnet_arm.main(["--out", str(cli_out), "--n", "1", "--size", "64"])
    stdout = capsys.readouterr().out
    assert "NAFNet arm: DONE" in stdout
    assert (cli_out / "nafnet_arm.json").is_file()


def test_blind_kernel_motion_refinement_and_guards() -> None:
    # Invalid inputs -> NOT_RUN
    for bad in (
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.complex64),
        np.full((64, 64, 3), np.nan, dtype=np.float32),
        np.full((64, 64, 3), 1, dtype=np.uint8),
    ):
        est, rec = blind_kernel.estimate_kernel(bad)
        assert est is None and rec["status"] == "NOT_RUN"

    # Normalized float [0, 1] with sharp multi-angle lines blurred by motion kernel
    sharp = np.full((180, 180, 3), 0.1, dtype=np.float32)
    cv2.rectangle(sharp, (25, 25), (155, 70), (0.95, 0.95, 0.95), -1)
    pts = np.array([[30, 150], [95, 85], [155, 145]], dtype=np.int32)
    cv2.fillPoly(sharp, [pts], (0.9, 0.9, 0.9))
    truth = motion_kernel(7, 30.0)
    blurred = cv2.filter2D(sharp, -1, truth)

    est, rec = blind_kernel.estimate_kernel(blurred)
    assert est is not None and rec["status"] == "RUN"
    assert rec["edge_count"] > 0

    # Direct local validation guards
    delta = np.zeros((3, 3), dtype=np.float32)
    delta[1, 1] = 1.0
    assert blind_kernel._valid_estimate(delta)[0] is None
    assert blind_kernel._valid_estimate(np.full((3, 3), -1.0))[0] is None
    assert blind_kernel._valid_estimate(np.zeros((3, 3), dtype=np.float32))[0] is None
    assert blind_kernel._valid_estimate(np.full((3, 3), np.nan))[0] is None

    valid = disk_kernel(2)
    with pytest.raises(ValueError, match="finite"):
        blind_kernel.kernel_error(np.full((3, 3), np.nan), valid)
    with pytest.raises(ValueError, match="negative"):
        blind_kernel.kernel_error(np.full((3, 3), -0.1), valid)
    with pytest.raises(ValueError, match="positive mass"):
        blind_kernel.kernel_error(np.zeros((3, 3)), valid)


def test_modelforge_error_bounds_shapes_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="quá nhỏ|move"):
        modelforge._rand_move_destination(rng, (32, 32), (4, 4, 24, 24))
    with pytest.raises(ValueError, match="quá nhỏ|splice"):
        modelforge._rand_splice_destination(rng, (32, 32), (4, 4, 24, 24), disjoint=True)
    with pytest.raises(ValueError, match="quá nhỏ"):
        modelforge._rand_splice_destination(rng, (20, 20), (4, 4, 16, 16), disjoint=False)

    poly = modelforge._shape_mask((64, 64), (8, 8, 24, 24), shape="polygon", rng=rng)
    assert int(poly.max()) == 255

    base = np.full((64, 64, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError, match="không hỗ trợ"):
        modelforge.forge(base, "bogus", modelforge.make_editor("classic"), rng)

    out_dir = tmp_path / "mf_cli"
    monkeypatch.setattr(
        sys,
        "argv",
        ["modelforge", "--n", "1", "--size", "128", "--backend", "classic", "--out", str(out_dir)],
    )
    modelforge.main()
    assert "backend=classic" in capsys.readouterr().out
    assert (out_dir / "meta.json").is_file()


def test_enhance_study_source_guard_estimated_not_run_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "source_set"
    source_dir.mkdir()
    with pytest.raises(ValueError, match="inside the source set"):
        enhance_study._assert_outside_source(source_dir / "sub" / "bad.png", source_dir)

    assert enhance_study._list_arg(" keypoint , block , ") == ["keypoint", "block"]

    # Flat sample makes blind_kernel return NOT_RUN when kernel_role == "estimated"
    flat = np.full((64, 64, 3), 120, dtype=np.uint8)
    for suf in ("_O.png", "_F.png"):
        (source_dir / f"easy_0000{suf}").write_bytes(cv2.imencode(".png", flat)[1].tobytes())
    (source_dir / "easy_0000_M.png").write_bytes(
        cv2.imencode(".png", np.zeros((64, 64), dtype=np.uint8))[1].tobytes()
    )
    (source_dir / "meta.json").write_text(json.dumps([{"id": "easy_0000", "seed": 1}]))

    out_dir = tmp_path / "study_out"
    out_dir.mkdir()
    monkeypatch.setattr(
        blind_kernel,
        "estimate_kernel",
        lambda _img: (None, {"status": "NOT_RUN", "reason": "insufficient straight-edge evidence"}),
    )
    enhance_study.main(
        [
            "--set",
            str(source_dir),
            "--out",
            f"{out_dir}/",
            "--degradations",
            "disk",
            "--restorers",
            "blind_lucy",
            "--methods",
            "ela",
        ]
    )
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[0].endswith("enhance_study.json")
    payload = json.loads((out_dir / "enhance_study.json").read_text(encoding="utf-8"))
    est_arms = [
        arm for arm in payload["samples"][0]["arms"] if arm.get("kernel_role") == "estimated"
    ]
    assert est_arms and est_arms[0]["status"] == "NOT_RUN"

    with pytest.raises(SystemExit) as exc:
        enhance_study.main(["--set", str(tmp_path / "nonexistent"), "--out", str(tmp_path / "x")])
    assert exc.value.code == 2


def test_derive_sibling_guards_and_json_safe(tmp_path: Path) -> None:
    src = tmp_path / "src.png"
    dst = tmp_path / "src.view.png"
    img = np.full((24, 24, 3), 100, dtype=np.uint8)
    src.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    sha = hashlib.sha256(src.read_bytes()).hexdigest()

    write_sibling(src, dst, img, "unsharp", {"ksize": 3}, sha, overwrite=False)
    with pytest.raises(ValueError, match="already exists"):
        write_sibling(src, dst, img, "unsharp", {"ksize": 3}, sha, overwrite=False)
    with pytest.raises(ValueError, match="differ from the source"):
        write_sibling(src, src, img, "unsharp", {"ksize": 3}, sha, overwrite=True)

    converted = _json_safe(
        {
            "arr": np.array([[1, 2]], dtype=np.uint8),
            "i": np.int64(7),
            "f": np.float32(2.5),
            "t": (1, Path("x/y")),
        }
    )
    encoded = json.dumps(converted)
    assert "[[1, 2]]" in encoded and "2.5" in encoded


def test_folder_ingestor_edge_branches_and_sibling_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    watch = tmp_path / "watch"
    data_dir = tmp_path / "data"
    watch.mkdir()

    good_img = np.full((64, 64, 3), 110, dtype=np.uint8)
    raw_png = cv2.imencode(".png", good_img)[1].tobytes()
    (watch / ".hidden.png").write_bytes(raw_png)
    (watch / "part.tmp").write_bytes(raw_png)
    (watch / "prior.view.png").write_bytes(raw_png)
    (watch / "not_an_image.png").write_bytes(b"NOT_IMAGE_BYTES_123456")

    a_png = watch / "a.png"
    b_dup_png = watch / "b_dup.png"
    a_png.write_bytes(raw_png)
    b_dup_png.write_bytes(raw_png)

    cfg = IngestConfig(
        data_dir=data_dir,
        methods=("ela",),
        max_pending=10,
        enhance_sibling=True,
    )
    svc = AnalysisService(load_thresholds())

    with pytest.raises(ValueError, match="settle_delay"):
        FolderIngestor(cfg, svc, "h", settle_delay=0.9)

    ing = FolderIngestor(cfg, svc, "cfg_hash_1", settle_delay=0.0)
    with pytest.raises(NotADirectoryError):
        ing.scan_once(tmp_path / "does_not_exist")

    # Mark c_symlink.png as symlink via helper
    sym_file = watch / "c_sym.png"
    sym_file.write_bytes(raw_png)
    real_is_link = __import__("imageguard.ingest.folder", fromlist=["_entry_is_link"])._entry_is_link
    monkeypatch.setattr(
        "imageguard.ingest.folder._entry_is_link",
        lambda p: p.name == "c_sym.png" or real_is_link(p),
    )

    summary = ing.scan_once(watch)
    assert summary.analyzed == 1
    assert summary.skipped_duplicate == 1
    assert summary.errors == 1  # symlink error
    assert (watch / "a.view.png").is_file()
    assert (watch / "a.view.json").is_file()

    # Re-scanning with identical symlink skips repeat error log
    summary2 = ing.scan_once(watch)
    assert not summary2.has_activity

    # Simulate service raising and SHA mismatch isolation
    d_err = watch / "d_err.png"
    d_err.write_bytes(cv2.imencode(".png", good_img + 5)[1].tobytes())
    e_sha = watch / "e_sha.png"
    e_sha.write_bytes(cv2.imencode(".png", good_img + 10)[1].tobytes())

    class _WeirdSvc:
        cfg = load_thresholds()

        def analyze(self, req):
            if req.path.name == "d_err.png":
                raise RuntimeError("boom")
            if req.path.name == "e_sha.png":
                return SimpleNamespace(image_sha256="0" * 64, report_id="r1")
            return ErrorResult(code=ErrorCode.INTERNAL, message="bad")

    ing_err = FolderIngestor(cfg, _WeirdSvc(), "cfg_hash_1", settle_delay=0.0)
    s_err = ing_err.scan_once(watch)
    assert s_err.errors == 2


def test_http_pull_validation_and_redirect_handling(tmp_path: Path) -> None:
    dest = tmp_path / "pulls"
    dest.mkdir()

    assert pull_due("not-a-dict", dest) == []  # type: ignore[arg-type]
    assert pull_due({"enabled": False, "urls": ["http://example.com/a.jpg"]}, dest) == []
    assert pull_due({"enabled": True, "urls": []}, dest) == []

    base_ok = {"enabled": True, "urls": ["http://example.com/a.jpg"], "interval_s": 5, "timeout_s": 5, "max_retries": 0, "hosts": ["example.com"]}
    assert pull_due({**base_ok, "interval_s": "bad"}, dest)[0]["reason"] == "invalid_interval"
    assert pull_due({**base_ok, "interval_s": 1}, dest)[0]["reason"] == "interval_floor"
    assert pull_due({**base_ok, "timeout_s": 99}, dest)[0]["reason"] == "invalid_timeout"
    assert pull_due({**base_ok, "max_retries": True}, dest)[0]["reason"] == "invalid_retries"
    assert pull_due({**base_ok, "hosts": ["example.com"]}, dest, max_bytes=0)[0]["reason"] == "invalid_max_bytes"
    assert pull_due({**base_ok, "hosts": "not-list"}, dest)[0]["reason"] == "invalid_allowlist"
    assert pull_due({**base_ok, "username_env": 123}, dest)[0]["reason"] == "credentials_invalid"
    assert pull_due({**base_ok, "secrets_file": "rel/secret.txt"}, dest)[0]["reason"] == "secrets_path_must_be_absolute"

    repo_root = Path(__file__).resolve().parents[1]
    assert pull_due({**base_ok, "secrets_file": str(repo_root / "pyproject.toml")}, dest)[0]["reason"] == "secrets_file_inside_repository"
    assert pull_due({**base_ok, "hosts": ["other.com"]}, dest)[0]["reason"] == "host_not_allowed"
    assert pull_due({**base_ok, "urls": ["http://169.254.169.254/a.jpg"], "hosts": ["169.254.169.254"]}, dest)[0]["reason"] == "metadata_blocked"
    assert pull_due({**base_ok, "urls": ["http://127.0.0.1/a.jpg"], "hosts": ["127.0.0.1"]}, dest)[0]["reason"] == "loopback_blocked"

    # Real local HTTP server testing 302 -> 200 JPEG, missing Location, redirect limit, and 404
    jpg_bytes = cv2.imencode(
        ".jpg", np.full((32, 32, 3), 120, dtype=np.uint8)
    )[1].tobytes()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/snap.jpg":
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg_bytes)))
                self.end_headers()
                self.wfile.write(jpg_bytes)
            elif self.path == "/redir_ok":
                self.send_response(302)
                self.send_header("Location", "/snap.jpg")
                self.end_headers()
            elif self.path == "/redir_noloc":
                self.send_response(302)
                self.end_headers()
            elif self.path.startswith("/loop"):
                self.send_response(302)
                self.send_header("Location", "/loop2")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        common = {
            "enabled": True,
            "hosts": ["127.0.0.1"],
            "interval_s": 5,
            "timeout_s": 5,
            "max_retries": 0,
            "allow_loopback": True,
        }
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        r_ok = pull_due({**common, "urls": [f"http://127.0.0.1:{port}/redir_ok"]}, dest, now=t0)
        assert r_ok[0]["status"] == "written"
        assert (dest / r_ok[0]["basename"]).is_file()

        r_noloc = pull_due(
            {**common, "urls": [f"http://127.0.0.1:{port}/redir_noloc"]},
            dest,
            now=t0 + timedelta(seconds=10),
        )
        assert r_noloc[0]["reason"] == "redirect_missing_location"

        r_lim = pull_due(
            {**common, "urls": [f"http://127.0.0.1:{port}/loop"]},
            dest,
            now=t0 + timedelta(seconds=20),
        )
        assert r_lim[0]["reason"] == "redirect_limit"

        r_404 = pull_due(
            {**common, "urls": [f"http://127.0.0.1:{port}/missing"]},
            dest,
            now=t0 + timedelta(seconds=30),
        )
        assert r_404[0]["reason"] == "http_status"
    finally:
        server.shutdown()
        server.server_close()


def test_cli_gui_importerror_eval_empty_methods_and_analyze_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def _deny_ui(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "imageguard.ui.app":
            raise ImportError("no qt")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _deny_ui)
    monkeypatch.setattr(sys, "argv", ["imageguard", "gui"])
    cli.main()
    assert "GUI khong kha dung" in capsys.readouterr().out
    monkeypatch.setattr(builtins, "__import__", real_import)

    monkeypatch.setattr(
        sys,
        "argv",
        ["imageguard", "eval", "--dataset", "synth", "--root", str(tmp_path), "--methods", ",,,"],
    )
    with pytest.raises(SystemExit) as exc_eval:
        cli.main()
    assert exc_eval.value.code == 2

    monkeypatch.setattr(
        sys,
        "argv",
        ["imageguard", "analyze", str(tmp_path / "absent.png"), "--no-persist"],
    )
    with pytest.raises(SystemExit) as exc_ana:
        cli.main()
    assert exc_ana.value.code == 1
    err_json = json.loads(capsys.readouterr().out)
    assert err_json["code"] == ErrorCode.PATH_INVALID.value
