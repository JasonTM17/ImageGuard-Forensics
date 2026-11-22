"""Pixel-level tests cho heatmap/mask/overlay, copy-move localization và ELA."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from imageguard.core.config import ThresholdSet, load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.io import safe_open
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.ela import ELA
from imageguard.forensics.block.pipeline import BlockCopyMove
from imageguard.forensics.keypoint.mask_builder import build_mask, upscale_mask
from imageguard.forensics.keypoint.pipeline import KeypointCopyMove
from imageguard.forensics.keypoint.viz import draw_pairs
from imageguard.forensics.localize.fusion_masks import fuse_masks
from imageguard.forensics.localize.heatmap import (
    normalize_map,
    overlay_heatmap,
    overlay_mask,
    render_heatmap,
)


def _doc(bgr: np.ndarray, name: str = "synth.png", fmt: str = "png") -> ImageDocument:
    h, w = bgr.shape[:2]
    return ImageDocument(
        array=bgr,
        width=w,
        height=h,
        channels=3,
        image_format=fmt,
        sha256="p" * 64,
        file_bytes=int(bgr.size),
        file_name=name,
    )


def _textured_copymove(size: int = 256) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    rng = np.random.default_rng(101)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[:, :, 0] = 115 + 45 * np.sin(xx / 7.0) * np.cos(yy / 8.5)
    base[:, :, 1] = 128 + 42 * np.cos(xx / 5.5) * np.sin(yy / 7.5)
    base[:, :, 2] = 120 + 40 * np.sin((xx - yy) / 6.0)
    base += rng.normal(0.0, 7.0, base.shape).astype(np.float32)
    img = np.clip(base, 0, 255).astype(np.uint8)
    # Rich corners/edges in source patch [32:100, 32:100].
    cv2.circle(img, (64, 64), 20, (235, 195, 65), -1)
    cv2.rectangle(img, (40, 44), (86, 88), (40, 75, 225), 2)
    cv2.line(img, (36, 92), (94, 38), (245, 245, 245), 2)

    src_box = (32, 32, 100, 100)  # (y0, x0, y1, x1)
    dst_box = (140, 140, 208, 208)
    img[dst_box[0] : dst_box[2], dst_box[1] : dst_box[3]] = img[
        src_box[0] : src_box[2], src_box[1] : src_box[3]
    ].copy()
    return img, src_box, dst_box


def test_heatmap_render_and_overlay_exact_pixel_math() -> None:
    rng = np.random.default_rng(23)
    bgr = rng.integers(10, 240, (40, 50, 3), dtype=np.uint8)
    orig = bgr.copy()

    # All-zero heat leaves every pixel of bgr untouched and does not mutate input.
    zero_heat = np.zeros((40, 50), dtype=np.float32)
    out_zero = overlay_heatmap(bgr, zero_heat, alpha=0.45)
    assert out_zero.shape == bgr.shape and out_zero.dtype == np.uint8
    np.testing.assert_array_equal(out_zero, bgr)
    np.testing.assert_array_equal(bgr, orig)

    # Known hot/cold pixels verify exact blending formula.
    heat = np.zeros((40, 50), dtype=np.float32)
    heat[10:20, 15:25] = 1.0
    heat[30:35, 35:45] = 0.5
    alpha = 0.4
    hm = render_heatmap(heat)
    assert hm.shape == bgr.shape and hm.dtype == np.uint8

    out = overlay_heatmap(bgr, heat, alpha=alpha)
    assert out.shape == bgr.shape and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    np.testing.assert_array_equal(bgr, orig)

    # Cold pixel stays exact.
    np.testing.assert_array_equal(out[0, 0], bgr[0, 0])
    # Hot pixel (norm=1.0 -> weight=0.4) matches formula.
    expected_hot = (bgr[12, 18].astype(np.float32) * (1.0 - alpha) + hm[12, 18].astype(np.float32) * alpha).astype(np.uint8)
    np.testing.assert_array_equal(out[12, 18], expected_hot)
    # Half pixel (norm=0.5 -> weight=0.2) matches formula.
    w_half = 0.5 * alpha
    expected_half = (bgr[32, 38].astype(np.float32) * (1.0 - w_half) + hm[32, 38].astype(np.float32) * w_half).astype(np.uint8)
    np.testing.assert_array_equal(out[32, 38], expected_half)

    # Non-matching heat shape resizes cleanly to bgr.shape[:2].
    small_heat = np.ones((10, 12), dtype=np.float32)
    out_resized = overlay_heatmap(bgr, small_heat, alpha=0.5)
    assert out_resized.shape == bgr.shape and out_resized.dtype == np.uint8
    assert float(normalize_map(small_heat).max()) == 1.0


def test_overlay_mask_exact_pixel_blend_and_custody() -> None:
    bgr = np.full((30, 30, 3), 100, dtype=np.uint8)
    orig = bgr.copy()
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    color = (0, 0, 255)
    alpha = 0.4

    out = overlay_mask(bgr, mask, color=color, alpha=alpha)
    np.testing.assert_array_equal(bgr, orig)
    assert out.shape == bgr.shape and out.dtype == np.uint8

    # Outside mask: unchanged.
    np.testing.assert_array_equal(out[0, 0], np.array([100, 100, 100], dtype=np.uint8))
    # Inside mask: cv2.addWeighted(100, 0.6, color, 0.4, 0) -> (60, 60, 162).
    expected_px = cv2.addWeighted(
        np.array([[[100, 100, 100]]], dtype=np.uint8),
        1 - alpha,
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        alpha,
        0,
    )[0, 0]
    np.testing.assert_array_equal(out[12, 12], expected_px)


def test_fuse_masks_resizes_mismatched_shapes_and_handles_modes() -> None:
    m1 = np.zeros((64, 64), dtype=np.uint8)
    m1[10:30, 10:30] = 255
    m2 = np.zeros((32, 32), dtype=np.uint8)
    m2[10:20, 10:20] = 255  # maps to [20:40, 20:40] at 64x64

    union = fuse_masks([m1, m2], mode="any", close_k=0)
    inter = fuse_masks([m1, m2], mode="all", close_k=0)
    assert union is not None and inter is not None
    assert union.shape == (64, 64) and inter.shape == (64, 64)
    assert union.dtype == np.uint8 and inter.dtype == np.uint8
    assert set(np.unique(union)).issubset({0, 255})
    assert set(np.unique(inter)).issubset({0, 255})
    # Overlap region [20:30, 20:30] is 255 in both union and intersection.
    assert np.all(inter[20:30, 20:30] == 255)
    assert np.all(inter[10:19, 10:19] == 0)
    assert np.all(union[10:20, 10:20] == 255)
    assert np.all(union[31:40, 31:40] == 255)


def test_draw_pairs_scales_coordinates_and_build_mask_bounds() -> None:
    base = np.full((64, 64, 3), 20, dtype=np.uint8)
    orig = base.copy()
    kps = [cv2.KeyPoint(10.0, 10.0, 4.0), cv2.KeyPoint(25.0, 20.0, 4.0)]
    pairs = np.array([[0, 1]], dtype=np.int32)
    inliers = np.array([0], dtype=np.int32)

    drawn = draw_pairs(base, kps, pairs, inliers, scale=0.5)
    np.testing.assert_array_equal(base, orig)
    # Keypoints scaled by 1 / 0.5 = 2.0 -> (20, 20) and (50, 40)
    assert not np.array_equal(drawn[20, 20], base[20, 20])
    assert not np.array_equal(drawn[40, 50], base[40, 50])

    mask, heat = build_mask(kps, pairs, inliers, (64, 64))
    assert mask.shape == (64, 64) and mask.dtype == np.uint8
    assert heat.shape == (64, 64) and heat.dtype == np.float32
    assert 0.0 <= float(heat.min()) <= float(heat.max()) <= 1.0
    assert set(np.unique(mask)).issubset({0, 255})

    up = upscale_mask(mask, 0.5, (128, 128))
    assert up.shape == (128, 128) and up.dtype == np.uint8
    assert set(np.unique(up)).issubset({0, 255})


def test_copymove_detectors_mark_known_regions_and_handle_downscale() -> None:
    img, src_box, dst_box = _textured_copymove(size=256)
    doc = _doc(img, "cm.png")
    ts = load_thresholds()

    kp_res = KeypointCopyMove().analyze(doc, ts)
    bl_res = BlockCopyMove().analyze(doc, ts)
    assert kp_res.verdict == Verdict.SUSPICIOUS_INDICATION
    assert bl_res.verdict == Verdict.SUSPICIOUS_INDICATION
    assert kp_res.mask is not None and kp_res.mask.shape == (256, 256)
    assert bl_res.mask is not None and bl_res.mask.shape == (256, 256)
    assert set(np.unique(kp_res.mask)).issubset({0, 255})
    assert set(np.unique(bl_res.mask)).issubset({0, 255})

    # Both detectors must place nonzero mask pixels inside source and destination boxes.
    for m in (kp_res.mask, bl_res.mask):
        assert np.any(m[src_box[0] : src_box[2], src_box[1] : src_box[3]] == 255)
        assert np.any(m[dst_box[0] : dst_box[2], dst_box[1] : dst_box[3]] == 255)

    # Downscaled working copy (scale < 1.0) rescales masks & heatmaps back to original dims.
    raw = dict(ts.raw)
    raw["limits"] = {**raw["limits"], "work_max_side": {"value": 192}}
    raw["block"] = {**raw["block"], "max_side": {"value": 192}}
    small_cfg = ThresholdSet(raw=raw)
    doc_scaled = _doc(img, "cm_scaled.png")

    kp_s = KeypointCopyMove().analyze(doc_scaled, small_cfg)
    bl_s = BlockCopyMove().analyze(doc_scaled, small_cfg)
    assert doc_scaled.resolution_reduced is True
    assert kp_s.heatmap is not None and kp_s.heatmap.shape == (256, 256)
    assert bl_s.heatmap is not None and bl_s.heatmap.shape == (256, 256)
    assert any("downscale" in lim for lim in kp_s.limitations)
    assert any("scale=" in lim for lim in bl_s.limitations)


def test_keypoint_high_coverage_downgrades_and_block_weak_signal_inconclusive(monkeypatch) -> None:
    img, _, _ = _textured_copymove(size=256)
    ts = load_thresholds()

    # Force build_mask to cover > 50% of image to verify the >50% natural-texture safeguard.
    from imageguard.forensics.keypoint import pipeline as kp_pipe

    def big_mask(_kps, _pairs, _inliers, shape):
        h, w = shape
        m = np.full((h, w), 255, dtype=np.uint8)
        h_map = np.ones((h, w), dtype=np.float32)
        return m, h_map

    monkeypatch.setattr(kp_pipe, "build_mask", big_mask)
    kp_res = KeypointCopyMove().analyze(_doc(img), ts)
    assert kp_res.verdict == Verdict.INCONCLUSIVE
    assert any("> 50%" in lim for lim in kp_res.limitations)

    # Block weak signal: 0 < frac < min_area_frac -> INCONCLUSIVE.
    raw = dict(ts.raw)
    raw["block"] = {**raw["block"], "min_area_frac": {"value": 0.99}}
    weak_cfg = ThresholdSet(raw=raw)
    bl_res = BlockCopyMove().analyze(_doc(img), weak_cfg)
    assert bl_res.verdict == Verdict.INCONCLUSIVE
    assert any("tín hiệu yếu" in lim for lim in bl_res.limitations)


def test_ela_pixel_difference_recompressed_vs_pasted_patch(tmp_path: Path) -> None:
    ts = load_thresholds()
    rng = np.random.default_rng(77)
    base = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)

    # Pure JPEG saved at q90 vs JPEG with an uncompressed high-frequency patch pasted in.
    clean_path = tmp_path / "clean.jpg"
    assert cv2.imwrite(str(clean_path), base, [cv2.IMWRITE_JPEG_QUALITY, 90])

    reopened = cv2.imread(str(clean_path))
    assert reopened is not None
    # Paste an un-recompressed noisy/sharp patch at [80:160, 80:160] then save at q98
    # (or q95) so the spliced block exhibits distinct quantization error compared to q90 background.
    patch = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    reopened[80:160, 80:160] = patch
    tampered_path = tmp_path / "tampered.jpg"
    assert cv2.imwrite(str(tampered_path), reopened, [cv2.IMWRITE_JPEG_QUALITY, 95])

    doc_tampered = safe_open(tampered_path, ts)
    assert isinstance(doc_tampered, ImageDocument)
    r_tampered = ELA().analyze(doc_tampered, ts)
    assert r_tampered.verdict != Verdict.SUSPICIOUS_INDICATION
    assert r_tampered.heatmap is not None and r_tampered.heatmap.shape == (256, 256)
    assert r_tampered.mask is not None and r_tampered.mask.shape == (256, 256)
    assert r_tampered.heatmap.dtype == np.float32
    assert 0.0 <= float(r_tampered.heatmap.min()) <= float(r_tampered.heatmap.max()) <= 1.0

    # Inside pasted patch must have higher mean ELA heatmap than outer background.
    inside_mean = float(r_tampered.heatmap[90:150, 90:150].mean())
    outside_mean = float(
        np.concatenate(
            [r_tampered.heatmap[:60, :].ravel(), r_tampered.heatmap[180:, :].ravel()]
        ).mean()
    )
    assert inside_mean > outside_mean
    # Mask pixels (heat >= 0.5) must be nonzero inside the pasted region.
    assert np.any(r_tampered.mask[80:160, 80:160] == 255)
