"""Tests cho modelforge — classic backend chạy luôn; LaMa skip nếu thiếu model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from imageguard.evaluation.modelforge import (
    KINDS,
    MODEL_PATH,
    ClassicEditor,
    LaMaEditor,
    forge,
)
from imageguard.evaluation.synthforge import _base_image


@pytest.fixture()
def base():
    rng = np.random.Generator(np.random.PCG64(3))
    return _base_image(rng, (256, 256))


@pytest.mark.parametrize("kind", KINDS)
def test_classic_forge_produces_gt_mask(base, kind):
    rng = np.random.Generator(np.random.PCG64(11))
    foreign = _base_image(rng, (256, 256))
    forged, mask, spec = forge(base, kind, ClassicEditor(), rng, foreign)
    assert forged.shape == base.shape
    assert mask.shape == base.shape[:2]
    gt_frac = float((mask > 0).mean())
    assert 0.005 < gt_frac < 0.35, f"{kind}: GT {gt_frac:.1%} bất thường"
    # Edit thật sự đổi pixel trong vùng GT
    diff = np.abs(forged.astype(int) - base.astype(int)).sum(axis=2)
    assert (diff[mask > 0] > 20).sum() > 50, f"{kind}: vùng GT không đổi"
    assert spec["kind"] == kind


def test_move_erases_source(base):
    """move: vùng gốc bị inpaint → pixel tại src khác base — twin bị phá."""
    rng = np.random.Generator(np.random.PCG64(5))
    forged, _, spec = forge(base, "move", ClassicEditor(), rng)
    sx, sy, sw, sh = spec["src"]
    src_orig = base[sy:sy + sh, sx:sx + sw]
    src_new = forged[sy:sy + sh, sx:sx + sw]
    assert np.abs(src_orig.astype(int) - src_new.astype(int)).mean() > 5


def test_move_ground_truth_matches_clone_and_inpaint_footprints(base):
    class RecordingEditor:
        def clone(self, patch, pmask, dst, center):
            self.patch_shape = patch.shape[:2]
            self.clone_mask = pmask.copy()
            self.center = center
            return dst

        def inpaint(self, bgr, mask):
            self.source_mask = mask.copy()
            return bgr

    editor = RecordingEditor()
    _, gt, spec = forge(
        base, "move", editor,
        np.random.Generator(np.random.PCG64(5)))
    dx, dy, dw, dh = spec["dst"]

    assert editor.patch_shape == (dh, dw)
    dest_gt = np.zeros_like(gt)
    x0 = editor.center[0] - dw // 2
    y0 = editor.center[1] - dh // 2
    dest_gt[y0:y0 + dh, x0:x0 + dw] = editor.clone_mask
    expected_gt = np.maximum(editor.source_mask, dest_gt)
    assert np.array_equal(gt, expected_gt)
    assert (dx, dy) == (x0, y0)


def test_forge_small_image_does_not_crash():
    """_rand_region quảng cáo h,w ≥ ~24px — patch nhỏ không được phá cv2.ellipse."""
    rng = np.random.Generator(np.random.PCG64(3))
    base = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    for kind in KINDS:
        for seed in range(6):
            forged, mask, _spec = forge(
                base, kind, ClassicEditor(),
                np.random.Generator(np.random.PCG64(seed)), base)
            assert forged.shape == base.shape
            assert mask.shape == base.shape[:2]
            assert (mask > 0).any(), f"{kind} seed={seed}: GT rỗng"


def test_forge_small_splice_keep_raises_not_hangs():
    """16×16 splice_keep: không thể chứa 2 vùng tách rời → ValueError nhanh,
    không treo. Subprocess + timeout để suite không bị wedge nếu vòng lặp
    vẫn vô hạn."""
    root = Path(__file__).resolve().parents[1]
    code = (
        "import numpy as np\n"
        "from imageguard.evaluation.modelforge import forge, ClassicEditor\n"
        "rng = np.random.Generator(np.random.PCG64(0))\n"
        "base = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)\n"
        "try:\n"
        "    forge(base, 'splice_keep', ClassicEditor(), "
        "np.random.Generator(np.random.PCG64(0)), base)\n"
        "except ValueError:\n"
        "    print('VALUEERROR')\n"
        "else:\n"
        "    print('NO_ERROR')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert "VALUEERROR" in proc.stdout
    assert "NO_ERROR" not in proc.stdout


def test_small_image_error_messages_do_not_promise_unusable_sizes():
    """move/splice_keep cần 2 vùng tách rời; _rand_region không được quảng
    cáo '≥ ~14px' — 16×16 vẫn không chứa được 2 vùng tách rời."""
    rng = np.random.Generator(np.random.PCG64(0))
    tiny = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        forge(tiny, "move", ClassicEditor(),
              np.random.Generator(np.random.PCG64(0)))
    move_msg = str(excinfo.value)
    assert "tách rời" in move_msg
    assert "~14" not in move_msg

    smaller = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo2:
        forge(smaller, "erase", ClassicEditor(),
              np.random.Generator(np.random.PCG64(0)))
    assert "~14" not in str(excinfo2.value)


@pytest.mark.parametrize("kind", ("splice_keep", "splice_foreign"))
def test_splice_ground_truth_matches_clone_mask(base, kind):
    """GT phải là đúng clone mask tại vị trí clone — không phải ellipse lớn hơn."""
    class RecordingEditor:
        def clone(self, patch, pmask, dst, center):
            self.clone_mask = pmask.copy()
            self.center = center
            return dst

    editor = RecordingEditor()
    _, gt, spec = forge(
        base, kind, editor, np.random.Generator(np.random.PCG64(5)))
    dx, dy, dw, dh = spec["dst"]
    dest_gt = np.zeros_like(gt)
    dest_gt[dy:dy + dh, dx:dx + dw] = editor.clone_mask
    assert np.array_equal(gt, dest_gt)


@pytest.mark.parametrize("kind", ("move", "splice_keep", "splice_foreign"))
def test_clone_mask_is_inset_ellipse(base, kind):
    """Clone mask phải là ellipse thụt 2px trên nền 0 — góc 0, tâm 255.
    np.full(..., 255) khiến _inset_ellipse vô nghĩa → mask thành rectangle."""
    class RecordingEditor:
        def clone(self, patch, pmask, dst, center):
            self.clone_mask = pmask.copy()
            return dst

        def inpaint(self, bgr, mask):
            return bgr

    editor = RecordingEditor()
    forge(base, kind, editor, np.random.Generator(np.random.PCG64(5)), base)
    pm = editor.clone_mask
    assert pm[0, 0] == 0, f"{kind}: góc clone mask phải 0"
    cy, cx = pm.shape[0] // 2, pm.shape[1] // 2
    assert pm[cy, cx] == 255, f"{kind}: tâm clone mask phải 255"


def test_generate_set_writes_images_to_non_ascii_dir(tmp_path):
    import json

    from imageguard.evaluation.modelforge import generate_set

    out = tmp_path / "bộ_ảnh"
    mf = generate_set(2, seed=1, out_dir=out, size=(128, 128),
                      backend="classic", kinds=("erase", "splice_keep"))
    for e in json.loads(mf.read_text(encoding="utf-8")):
        stem = e["id"]
        ext = ".jpg" if e.get("format") == "jpeg" else ".png"
        for suf in ("_O.png", f"_F{ext}", "_M.png"):
            assert (out / f"{stem}{suf}").is_file(), f"thiếu {stem}{suf}"


def test_generate_set_contract(tmp_path):
    from imageguard.evaluation.modelforge import generate_set

    mf = generate_set(2, seed=1, out_dir=tmp_path / "mf",
                      size=(256, 256), backend="classic",
                      kinds=("erase", "splice_keep"))
    import json
    entries = json.loads(mf.read_text())
    assert len(entries) == 2
    for e in entries:
        stem = e["id"]
        ext = ".jpg" if e.get("format") == "jpeg" else ".png"
        for suf in ("_O.png", f"_F{ext}", "_M.png"):
            assert (mf.parent / f"{stem}{suf}").exists(), f"thiếu {stem}{suf}"


@pytest.mark.skipif(not MODEL_PATH.exists(),
                    reason="lama_fp32.onnx chưa tải (docs/modelforge.md)")
def test_lama_inpaint_fills_only_mask(base):
    mask = np.zeros((256, 256), np.uint8)
    cv2.ellipse(mask, (150, 150), (40, 30), 0, 0, 360, 255, -1)
    out = LaMaEditor().inpaint(base, mask)
    diff = np.abs(out.astype(int) - base.astype(int)).sum(axis=2)
    assert (diff[mask > 0] > 20).sum() > 100   # trong mask: nội dung mới
    assert diff[mask == 0].mean() < 5          # ngoài mask: gần như nguyên
