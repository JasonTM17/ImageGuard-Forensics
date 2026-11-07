"""Runtime-aware config, writable-data, and optional-model paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imageguard.core import paths
from imageguard.enhancement import nafnet


def _set_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_file: Path,
    frozen: bool,
    executable: Path | None = None,
    meipass: Path | None = None,
    compiled: object | None = None,
) -> None:
    monkeypatch.setattr(paths, "__file__", str(module_file))
    monkeypatch.setattr(paths, "FROZEN", frozen)
    if executable is None:
        monkeypatch.delattr(paths.sys, "executable", raising=False)
    else:
        monkeypatch.setattr(paths.sys, "executable", str(executable))
    if meipass is None:
        monkeypatch.delattr(paths.sys, "_MEIPASS", raising=False)
    else:
        monkeypatch.setattr(paths.sys, "_MEIPASS", str(meipass), raising=False)
    if compiled is None:
        monkeypatch.delattr(paths, "__compiled__", raising=False)
    else:
        monkeypatch.setattr(paths, "__compiled__", compiled, raising=False)


def test_source_paths_keep_repo_config_and_cwd_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "source tree æ字"
    module_file = repo / "src" / "imageguard" / "core" / "paths.py"
    foreign_cwd = tmp_path / "another project 另"
    (repo / "configs").mkdir(parents=True)
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    _set_runtime(monkeypatch, module_file=module_file, frozen=False)

    assert paths.config_path() == repo / "configs" / "thresholds.yaml"
    assert paths.data_root() == Path("data")
    assert paths.model_root() == repo


@pytest.mark.parametrize("layout", ["onedir", "onefile"])
def test_pyinstaller_paths_keep_existing_bundle_and_exe_roots(
    layout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_dir = tmp_path / "ImageGuard 影像 app"
    if layout == "onedir":
        meipass = app_dir / "_internal"
    else:
        meipass = tmp_path / "_MEI temporary 展开"
    module_file = meipass / "imageguard" / "core" / "paths.py"
    expected_model_root = module_file.resolve().parents[3]
    executable = app_dir / "ImageGuard.exe"
    foreign_cwd = tmp_path / "working directory 外部"
    (meipass / "configs").mkdir(parents=True)
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    _set_runtime(
        monkeypatch,
        module_file=module_file,
        frozen=True,
        executable=executable,
        meipass=meipass,
    )

    assert paths.config_path() == meipass / "configs" / "thresholds.yaml"
    assert paths.data_root() == app_dir / "data"
    assert paths.model_root() == expected_model_root


def test_nuitka_standalone_uses_dist_root_not_containing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = tmp_path / "Nuitka build 生成"
    dist_dir = build_dir / "ImageGuard standalone Æ"
    module_file = dist_dir / "imageguard" / "core" / "paths.py"
    executable = dist_dir / "python.exe"
    foreign_cwd = tmp_path / "launch cwd 外部"
    config = dist_dir / "configs" / "thresholds.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("thresholds: {}\n", encoding="utf-8")
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    compiled = SimpleNamespace(containing_dir=str(build_dir))
    _set_runtime(
        monkeypatch,
        module_file=module_file,
        frozen=False,
        executable=executable,
        compiled=compiled,
    )

    assert paths.config_path() == config
    assert paths.config_path().is_file()
    assert paths.data_root() == dist_dir / "data"
    assert paths.model_root() == dist_dir


def test_nafnet_model_path_is_derived_from_shared_model_root() -> None:
    assert nafnet.MODEL_PATH == (
        paths.model_root() / "models" / "deblurring_nafnet_2025may.onnx"
    )
