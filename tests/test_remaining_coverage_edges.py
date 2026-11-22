"""Coverage & hardening tests cho heatmap sanitization, __main__, ingest watch, NAFNet & Haar."""

from __future__ import annotations

import builtins
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from imageguard.enhancement import nafnet as nafnet_mod
from imageguard.forensics.localize.heatmap import normalize_map, overlay_heatmap
from imageguard.ingest import commands as ingest_cmd_mod
from imageguard.ipkit.features_basic import (
    haar_feature,
    haar_feature_map,
    integral_image,
)


def test_normalize_map_and_overlay_sanitize_nan_inf_and_negative() -> None:
    raw = np.array([[np.nan, -5.0], [np.inf, 4.0]], dtype=np.float32)
    norm = normalize_map(raw)
    assert norm.dtype == np.float32
    assert np.all(np.isfinite(norm))
    assert float(norm.min()) == 0.0
    assert float(norm.max()) == 1.0
    assert norm[0, 0] == 0.0 and norm[0, 1] == 0.0 and norm[1, 0] == 0.0 and norm[1, 1] == 1.0

    # All-invalid / negative evidence yields zeros and leaves overlay identical to base BGR.
    bgr = np.full((8, 8, 3), 90, dtype=np.uint8)
    bad_heat = np.full((8, 8), np.nan, dtype=np.float32)
    bad_heat[0, 0] = -10.0
    bad_heat[1, 1] = -np.inf
    out = overlay_heatmap(bgr, bad_heat, alpha=0.5)
    assert out.dtype == np.uint8 and out.shape == bgr.shape
    np.testing.assert_array_equal(out, bgr)


def test_python_module_main_entrypoint_invokes_cli_main(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("imageguard.cli.main", lambda: called.append(True))
    runpy.run_module("imageguard", run_name="__main__")
    assert called == [True]


def test_run_ingest_watch_mode_and_error_branches(tmp_path, monkeypatch, capsys) -> None:
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    monkeypatch.setattr(ingest_cmd_mod.time, "sleep", lambda _s: None)

    class _FakeIngestor:
        def __init__(self, *_a, **_kw) -> None:
            self.calls = 0

        def scan_once(self, _d):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    has_activity=True, analyzed=1, errors=0, skipped_duplicate=0, deferred=0
                )
            raise KeyboardInterrupt

    monkeypatch.setattr(ingest_cmd_mod, "FolderIngestor", _FakeIngestor)
    rc = ingest_cmd_mod.run_ingest("watch", watch_dir)
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingest watch: analyzed=1" in out

    # Watch exception -> exit 1
    class _FailWatchIngestor:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def scan_once(self, _d):
            raise RuntimeError("boom watch")

    monkeypatch.setattr(ingest_cmd_mod, "FolderIngestor", _FailWatchIngestor)
    assert ingest_cmd_mod.run_ingest("watch", watch_dir) == 1
    assert "ingest watch failed" in capsys.readouterr().err

    # Once exception -> exit 1
    assert ingest_cmd_mod.run_ingest("once", watch_dir) == 1
    assert "ingest scan failed" in capsys.readouterr().err


def test_nafnet_load_session_and_infer_validation(tmp_path, monkeypatch) -> None:
    nafnet_mod._load_session.cache_clear()
    weight = tmp_path / "model.onnx"
    weight.write_bytes(b"small")

    with pytest.raises(ValueError, match="unexpected size"):
        nafnet_mod._load_session(str(weight), 1, 5)

    monkeypatch.setattr(nafnet_mod, "MODEL_BYTES", 5)
    with pytest.raises(ValueError, match="SHA-256"):
        nafnet_mod._load_session(str(weight), 1, 5)

    # Mock valid SHA-256 and fake onnxruntime where use_deterministic_compute raises AttributeError
    digest = nafnet_mod._sha256_file(weight)
    monkeypatch.setattr(nafnet_mod, "MODEL_SHA256", digest)

    class _Opts:
        def __setattr__(self, name, value):
            if name == "use_deterministic_compute":
                raise AttributeError("unsupported")
            super().__setattr__(name, value)

    class _FakeSess:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, _out, feed):
            arr = next(iter(feed.values()))
            # Return mismatched spatial height to hit spatial mismatch check in infer()
            b, c, h, w = arr.shape
            return [np.zeros((b, c, h - 1, w), dtype=np.float32)]

    fake_ort = SimpleNamespace(
        SessionOptions=_Opts,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        InferenceSession=lambda *a, **kw: _FakeSess(),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    _sess, sha_out, det = nafnet_mod._load_session(str(weight), 2, 5)
    assert sha_out == digest
    assert det is False

    # Missing onnxruntime ImportError path
    nafnet_mod._load_session.cache_clear()
    real_import = builtins.__import__

    def no_ort(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("missing ort")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ort)
    with pytest.raises(ImportError, match="onnxruntime"):
        nafnet_mod._load_session(str(weight), 3, 5)
    monkeypatch.setattr(builtins, "__import__", real_import)

    # infer() input validation and spatial mismatch from mocked session
    with pytest.raises(ValueError, match="three-channel BGR"):
        nafnet_mod.infer(np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        nafnet_mod.infer(np.zeros((16, 16, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        nafnet_mod.infer(np.zeros((0, 16, 3), dtype=np.uint8))

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(nafnet_mod, "_session", lambda: (_FakeSess(), digest, False))
    with pytest.raises(ValueError, match="unexpected spatial size"):
        nafnet_mod.infer(np.zeros((64, 64, 3), dtype=np.uint8))
    nafnet_mod._load_session.cache_clear()


def test_ipkit_haar_feature_all_kinds_and_validation() -> None:
    gray = np.arange(36, dtype=np.uint8).reshape(6, 6)
    sat = integral_image(gray)

    for kind in ("two_h", "two_v", "three_h", "three_v", "four"):
        val = haar_feature(sat, kind, 0, 0, 6, 6)
        assert isinstance(val, float)

    with pytest.raises(ValueError, match="kind"):
        haar_feature(sat, "bogus", 0, 0, 4, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        haar_feature(sat, "two_h", 0, 0, 0, 4)
    with pytest.raises(ValueError, match="two_h"):
        haar_feature(sat, "two_h", 0, 0, 4, 1)
    with pytest.raises(ValueError, match="two_v"):
        haar_feature(sat, "two_v", 0, 0, 1, 4)
    with pytest.raises(ValueError, match="three_h"):
        haar_feature(sat, "three_h", 0, 0, 4, 2)
    with pytest.raises(ValueError, match="three_v"):
        haar_feature(sat, "three_v", 0, 0, 2, 4)
    with pytest.raises(ValueError, match="four"):
        haar_feature(sat, "four", 0, 0, 1, 4)

    with pytest.raises(TypeError, match="stride"):
        haar_feature_map(gray, "two_h", 2, 2, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stride"):
        haar_feature_map(gray, "two_h", 2, 2, 0)
    with pytest.raises(ValueError, match="win_h"):
        haar_feature_map(gray, "two_h", 0, 2, 1)
    with pytest.raises(ValueError, match="larger"):
        haar_feature_map(gray, "two_h", 8, 2, 1)
