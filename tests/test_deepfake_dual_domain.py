"""Regressions: Dual-Camera per-sensor domain isolation (hợp đồng V2).

V1 defect (arbiter P2, `dual-block-boundary-controller-repro.json`): lưới block
toàn khung bị cắt tại ``bw // 2`` trong khi ranh giới cảm biến là ``w // 2``
pixel → block hỗn hợp 2 cảm biến được chấp nhận làm nền tham chiếu; đồng thời
medianBlur / LBP / Otsu tính toàn cục nên halo bộ lọc vẫn rò nhiễu qua biên
ngay cả khi split khớp lưới block.

Hợp đồng repair được chấp nhận (controller implement, tests này chỉ khóa):
- heuristic hình học & luật chọn nửa giữ nguyên;
- lưới block mỗi nửa gồm block hoàn chỉnh của nửa đó, ROI clip trong tọa độ local;
- sau khi chọn, MỌI mảng cue (σ, LBP/micro-range, seam/nền, Otsu, FFT, HOG)
  chỉ lấy từ nửa đã chọn;
- cue block map local map về pixel-rect nửa đó, zero-pad full working frame,
  upscale về doc; ``roi_rect`` vẫn tọa độ global;
- downscale resize độc lập từng nửa nguồn;
- đường số học ảnh đơn giữ nguyên.

Kỳ vọng: RED trên integration v1 (sha256 deepfake_cues.py
3e63b7e907589110d8c8df6104335e0ace8ea7a6330c4693eb63f5f1316eb88e),
GREEN sau khi controller sửa. Các control đơn ảnh/nền hợp lệ phải GREEN cả 2.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import imageguard.forensics.auxiliary.deepfake_cues as dc_mod
from imageguard.core.config import load_thresholds
from imageguard.core.enums import Verdict
from imageguard.core.models import ImageDocument
from imageguard.forensics.auxiliary.deepfake_cues import DeepfakeCues

_H = 128
_BS = 16  # noise.block trong thresholds.yaml → đổi config thì đổi expected grid.


def _doc(arr: np.ndarray, name: str = "dual.png") -> ImageDocument:
    h, w = arr.shape[:2]
    return ImageDocument(
        array=arr,
        width=w,
        height=h,
        channels=3,
        image_format="png",
        sha256="d" * 64,
        file_bytes=int(arr.size),
        file_name=name,
    )


def _noise_half(h: int, w: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(120.0 + rng.normal(0.0, sigma, (h, w, 3)), 0, 255).astype(np.uint8)


def _dual_frame(w: int, left_sigma: float, right_sigma: float, seed: int = 42) -> np.ndarray:
    mid = w // 2
    return np.hstack(
        [_noise_half(_H, mid, left_sigma, seed), _noise_half(_H, w - mid, right_sigma, seed + 1)]
    )


class _SeqFaceROI:
    """_detect_face_roi giả lập: trả (rect, face, src) theo thứ tự gọi trái→phải."""

    def __init__(self, returns: list[tuple[tuple[int, int, int, int], bool, str]]) -> None:
        self._returns = list(returns)
        self.calls = 0

    def __call__(self, work_arr, gray_u8, block_size):
        r = self._returns[self.calls % len(self._returns)]
        self.calls += 1
        return r


def _force_side(monkeypatch: pytest.MonkeyPatch, rois: list, zero_drop: bool = True) -> None:
    """Khóa ROI + phía chọn deterministic qua roi/face flags, KHÔNG mock cue đã chọn.

    ``zero_drop`` chỉ triệt tiêu đầu đo *lựa chọn nửa* (drop_l/drop_r) — y hệt
    spy trong test_deepfake_background_isolation hiện hữu — để việc nhiễu nửa
    kia không được lật selection. Các cue pipeline của nửa đã chọn chạy code thật.
    """
    if zero_drop:
        real_stats = dc_mod._roi_sigma_stats

        def _zero_drop(sigma_region, rect, bs_eff, *a, **k):
            med_in, med_bg, _ = real_stats(sigma_region, rect, bs_eff)
            return med_in, med_bg, 0.0

        monkeypatch.setattr(dc_mod, "_roi_sigma_stats", _zero_drop)
    monkeypatch.setattr(dc_mod, "_detect_face_roi", _SeqFaceROI(rois))


def _selected_evidence(res, opposite: str) -> dict:
    """Toàn bộ evidence trừ telemetry σ của nửa đối diện (được phép đổi)."""
    return {k: v for k, v in res.evidence.items() if k != f"dual_sigma_{opposite}"}


def _result_snapshot(res) -> dict:
    return {
        "verdict": res.verdict,
        "score": res.score,
        "mask": None if res.mask is None else res.mask.copy(),
        "heatmap": None if res.heatmap is None else res.heatmap.copy(),
    }


def _assert_identical(a: dict, b: dict, ctx: str) -> None:
    assert a["evidence"] == b["evidence"], f"{ctx}: evidence của nửa đã chọn đổi khi nhiễu camera kia"
    assert a["verdict"] is b["verdict"], f"{ctx}: verdict flip {a['verdict']} -> {b['verdict']}"
    assert a["score"] == b["score"], f"{ctx}: score {a['score']} != {b['score']}"
    if a["mask"] is None or b["mask"] is None:
        assert a["mask"] is None and b["mask"] is None, f"{ctx}: mask xuất hiện/biến mất"
    else:
        assert np.array_equal(a["mask"], b["mask"]), f"{ctx}: mask đổi"
    if a["heatmap"] is not None and b["heatmap"] is not None:
        assert np.array_equal(a["heatmap"], b["heatmap"]), f"{ctx}: heatmap đổi"
    else:
        assert a["heatmap"] is None and b["heatmap"] is None, f"{ctx}: heatmap xuất hiện/biến mất"


@pytest.mark.parametrize("w", [600, 601, 603])
def test_dual_right_selected_other_sensor_perturbation_is_invisible_unaligned(
    monkeypatch: pytest.MonkeyPatch, w: int
) -> None:
    """RED(v1): split không khớp block (600) & odd width — block hỗn hợp rò σ nền.

    Với selection phải bị khóa, thay đổi CHỈ camera trái phải giữ mọi đại lượng
    của nửa phải (σ_out, seam, LBP, Otsu blocks, FFT/HOG, score, verdict, mask,
    heatmap) nguyên vẹn. V1 outer_bg gồm col 18 = block hỗn hợp (288..303) → đỏ.
    """
    rois = [
        ((30, 30, 60, 60), False, "ctrl_left"),
        # local: ROI chiếm ~200px nửa phải → outer_bg = {col 18 (mixed), 34..36}
        ((20, 16, 200, 96), True, "ctrl_right_face"),
    ]
    _force_side(monkeypatch, rois)
    cfg = load_thresholds()
    snaps = []
    for left_sigma in (0.0, 90.0):
        frame = _dual_frame(w, left_sigma=left_sigma, right_sigma=4.0)
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_roi_side"] == "right"
        snap = _result_snapshot(res)
        snap["evidence"] = _selected_evidence(res, opposite="left")
        snaps.append(snap)
    _assert_identical(snaps[0], snaps[1], f"w={w} right-selected")


def test_dual_right_selected_other_sensor_halo_leaks_at_aligned_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED(v1): split 640 khớp lưới block — vẫn rò qua halo medianBlur/LBP/Otsu.

    Không có block hỗn hợp ở 640 (mid_bx*16 = mid_x = 320), nhưng σ block ranh
    giới (px 320..335) chứa residual của px 320-321 — cửa sổ medianBlur k=5 lấy
    px 318-319 bên trái; LBP/micro_range tương tự. outer_bg = {20,21,22} chứa
    block 20 bị nhiễm → med_out/flat_out/Otsu/score đổi theo camera trái.
    """
    w = 640
    _force_side(
        monkeypatch,
        [
            ((30, 30, 60, 60), False, "ctrl_left"),
            # global rx = 320+64 = 384 → inside 24..38; outer_bg = {20,21,22}.
            ((64, 0, 240, 128), True, "ctrl_right_face"),
        ],
    )
    cfg = load_thresholds()
    snaps = []
    for left_sigma in (0.0, 90.0):
        frame = _dual_frame(w, left_sigma=left_sigma, right_sigma=4.0)
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_roi_side"] == "right"
        snap = _result_snapshot(res)
        snap["evidence"] = _selected_evidence(res, opposite="left")
        snaps.append(snap)
    # kept_blocks nằm trong evidence → Otsu/thành phần liên thông cũng bị khóa.
    _assert_identical(snaps[0], snaps[1], "w=640 aligned right-selected")


def test_dual_left_selected_roi_crossing_split_is_clipped_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED(v1): ROI trái được clip tại ranh giới cảm biến, cue không lấn nửa phải.

    ROI trái cố tình vượt mid_x (250..349). V1: inside_roi tràn sang block của
    nửa phải và ``roi_f``/LBP lấy pixel phải → nhiễu camera phải đổi cue trái.
    V2: clip trong local coords → roi_rect global kết thúc ≤ mid_x.
    """
    w = 600
    mid = w // 2
    _force_side(
        monkeypatch,
        [
            ((250, 16, 80, 96), True, "ctrl_left_face"),  # tràn sang x>300
            ((30, 30, 60, 60), False, "ctrl_right"),
        ],
    )
    cfg = load_thresholds()
    snaps = []
    for right_sigma in (0.0, 90.0):
        frame = _dual_frame(w, left_sigma=4.0, right_sigma=right_sigma)
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_roi_side"] == "left"
        rx, _ry, rw, _rh = res.evidence["roi_rect"]
        assert rx + rw <= mid, f"roi_rect tràn ranh giới cảm biến: {[rx, _ry, rw, _rh]}"
        snap = _result_snapshot(res)
        snap["evidence"] = _selected_evidence(res, opposite="right")
        snaps.append(snap)
    _assert_identical(snaps[0], snaps[1], "w=600 left-selected crossing ROI")


def test_roi_sigma_stats_receives_per_half_complete_block_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED(v1): _roi_sigma_stats phải nhận lưới block HOÀN CHỈNH của từng nửa.

    Spy gọi implementation thật, ghi (shape region, rect). Với w=600, mỗi nửa
    300px chứa đúng 300//16 = 18 block hoàn chỉnh; V1 đưa nửa phải 19 cột
    (gồm block hỗn hợp px 288..303 chứa 12 px camera trái).
    """
    calls: list[tuple[tuple[int, ...], tuple[int, int, int, int], int]] = []
    real_stats = dc_mod._roi_sigma_stats

    def spy(sigma_region, rect, bs_eff, *a, **k):
        calls.append((tuple(sigma_region.shape), tuple(rect), int(bs_eff)))
        return real_stats(sigma_region, rect, bs_eff)

    monkeypatch.setattr(dc_mod, "_roi_sigma_stats", spy)
    monkeypatch.setattr(
        dc_mod,
        "_detect_face_roi",
        _SeqFaceROI(
            [
                ((0, 0, 300, 128), False, "forced"),
                ((4, 0, 296, 128), True, "forced"),
            ]
        ),
    )
    frame = _dual_frame(600, left_sigma=0.0, right_sigma=4.0)
    DeepfakeCues().analyze(_doc(frame), load_thresholds())

    assert len(calls) == 2, f"expected per-half selection probes, got {calls}"
    half_w = 300
    for idx, (shape, rect, bs_eff) in enumerate(calls):
        side = "left" if idx == 0 else "right"
        rx, _ry, rw, _rh = rect
        assert 0 <= rx and rx + rw <= half_w, f"{side} rect {rect} không phải local-coords nửa {half_w}px"
        complete = half_w // bs_eff
        assert shape[1] == complete, (
            f"{side} grid {shape[1]} cột ≠ {complete} block hoàn chỉnh — "
            "có cột hỗn hợp lấn ranh giới cảm biến"
        )
        assert shape[1] * bs_eff <= half_w


@pytest.mark.parametrize("w", [601])
def test_dual_heatmap_zero_outside_selected_half_odd_width(
    monkeypatch: pytest.MonkeyPatch, w: int
) -> None:
    """RED(v1): heatmap phải zero-pad ngoài nửa đã chọn (odd width 601).

    Camera trái yên tĩnh hơn nền phải → sigma_drop trái > 0 → V1 tô nóng cả nửa
    ngoại cảm biến. V2 map cue local → pixel-rect nửa phải, zero ngoài nửa đó.
    """
    mid = w // 2
    _force_side(
        monkeypatch,
        [
            ((30, 30, 60, 60), False, "ctrl_left"),
            ((20, 16, 200, 96), True, "ctrl_right_face"),
        ],
    )
    frame = _dual_frame(w, left_sigma=1.0, right_sigma=30.0)
    # Vùng mặt mịn hơn nền → heat dương thật trong nửa phải (chống assert rỗng).
    frame[16:112, mid + 20 : mid + 220] = _noise_half(96, 200, 2.0, seed=7)
    res = DeepfakeCues().analyze(_doc(frame), load_thresholds())

    assert res.evidence["dual_roi_side"] == "right"
    assert res.heatmap is not None and res.heatmap.shape[:2] == (_H, w)
    left_zone = res.heatmap[:, : mid - 4]
    right_zone = res.heatmap[:, mid + 4 :]
    assert float(np.max(left_zone)) == 0.0, (
        f"heatmap rò sang nửa camera trái (max={float(np.max(left_zone)):.4f})"
    )
    assert float(np.max(right_zone)) > 0.0, "heatmap rỗng cả nửa đã chọn — test rỗng"
    if res.mask is not None:
        assert not np.any(res.mask[:, :mid]), "mask rò sang nửa camera kia"


def test_dual_downscaled_frame_keeps_per_half_resize_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED(v1): đường downscale phải resize TỪNG nửa nguồn độc lập.

    aux_max_side=512 (override in-test, không sửa config) → doc 240×600 scale
    ~0.85. V1 resize cả khung: interpolation lấy pixel camera kia ở biên; lưới
    block toàn cục vẫn chứa cột hỗn hợp → nhiễu trái đổi cue phải đã chọn.
    """
    cfg = load_thresholds()
    limits = cfg.raw.setdefault("limits", {})
    limits["aux_max_side"] = {"value": 512}  # derived ThresholdSet in-test only.
    _force_side(
        monkeypatch,
        [
            ((30, 30, 60, 60), False, "ctrl_left"),
            ((20, 16, 200, 96), True, "ctrl_right_face"),  # local coords nửa work
        ],
    )
    snaps = []
    for left_sigma in (0.0, 90.0):
        mid = 300
        frame = np.hstack(
            [
                _noise_half(240, mid, left_sigma, 42),
                _noise_half(240, 600 - mid, 4.0, 43),
            ]
        )
        res = DeepfakeCues().analyze(_doc(frame), cfg)
        assert res.evidence["dual_camera_mode"] is True
        assert res.evidence["dual_roi_side"] == "right"
        assert res.evidence["scale"] < 1.0, "downscale path không kích hoạt"
        snap = _result_snapshot(res)
        snap["evidence"] = _selected_evidence(res, opposite="left")
        snaps.append(snap)
    _assert_identical(snaps[0], snaps[1], "downscaled w=600 right-selected")


def test_dual_right_valid_background_still_computes_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control GREEN cả 2: nửa phải có mặt + nền hợp lệ → pipeline tính bình thường."""
    _force_side(
        monkeypatch,
        [
            ((40, 40, 60, 60), False, "ctrl_left"),
            ((60, 50, 140, 120), True, "ctrl_right_face"),
        ],
    )
    frame = _dual_frame(640, left_sigma=8.0, right_sigma=8.0)
    res = DeepfakeCues().analyze(_doc(frame), load_thresholds())

    assert res.evidence["dual_camera_mode"] is True
    assert res.evidence["dual_roi_side"] == "right"
    assert res.score is not None
    assert "sigma_out_median" in res.evidence
    assert res.verdict in {Verdict.INCONCLUSIVE, Verdict.NO_INDICATION}


def test_single_image_numeric_baseline_matches_v1_snapshot() -> None:
    """Control GREEN cả 2: đường ảnh đơn phải giữ nguyên số học V1.

    Golden values CHỤP TỪ chạy thật trên integration v1 (deepfake_cues.py
    sha256 3e63b7e907589110d8c8df6104335e0ace8ea7a6330c4693eb63f5f1316eb88e,
    `main .venv`, ngày 2026-11-18) — không phải oracle tự chế. Cố định cả
    evidence/score/verdict và digest mask+heatmap (trừ duration_ms).
    """
    rng = np.random.default_rng(9)
    img = np.clip(128 + rng.normal(0.0, 16.0, (256, 256, 3)), 0, 255).astype(np.uint8)
    res = DeepfakeCues().analyze(_doc(img, "single.png"), load_thresholds())

    expected_evidence = dict(_SINGLE_V1_EVIDENCE)
    assert res.evidence == expected_evidence
    assert res.verdict is _SINGLE_V1_VERDICT
    assert res.score == _SINGLE_V1_SCORE
    assert (res.mask is None) is _SINGLE_V1_MASK_NONE
    if res.mask is not None:
        assert _sha(res.mask) == _SINGLE_V1_MASK_SHA
    assert _sha(res.heatmap) == _SINGLE_V1_HEAT_SHA


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --- Golden V1 (chụp từ probe run thật — xem docstring test trên) -----------
_SINGLE_V1_VERDICT = Verdict.NO_INDICATION
_SINGLE_V1_SCORE = 0.0071
_SINGLE_V1_MASK_NONE = True
_SINGLE_V1_MASK_SHA = ""
_SINGLE_V1_HEAT_SHA = "dc382a7e201e9afd7bdad9615c9ace22eab057ec0d2a6766b7274a6e059e1373"
_SINGLE_V1_EVIDENCE: dict = {
    "variant": "dip_4pillar_deepfake_cues",
    "roi_source": "central_fallback",
    "dual_camera_mode": False,
    "dual_sigma_left": None,
    "dual_sigma_right": None,
    "dual_roi_side": None,
    "face_detected": False,
    "corroborated": False,
    "roi_rect": [64, 64, 128, 128],
    "sigma_in_median": 10.683,
    "sigma_out_median": 10.501,
    "smooth_cue": 0.0,
    "seam_cue": 0.1079,
    "lbp_cue": 0.0,
    "lbp_chi2": 0.0003,
    "lbp_flat_in": 0.2227,
    "lbp_flat_out": 0.2227,
    "freq_cue": 0.0,
    "fft_hf_peak_ratio": 2.957,
    "spectral_rolloff": 0.992,
    "seam_hough_cue": 0.0,
    "seam_hough_lines": 0,
    "hog_asym": 0.1783,
    "kept_blocks": 0,
    "block": 16,
    "scale": 1.0,
}
