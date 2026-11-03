"""ModelForge — sinh ảnh chỉnh sửa bằng MODEL LOCAL (không API, không cloud).

Backend:
- `lama`   : LaMa ONNX (advimman/lama, Apache-2.0) qua onnxruntime — ML
  inpainting thật: Fourier convolutions nhìn toàn cảnh, lấp vùng mask theo
  cấu trúc xung quanh. Model `models/lama_fp32.onnx` (208MB, không commit,
  sha256 pin trong docs).
- `classic`: cv2.inpaint (Telea) + seamlessClone — thuật toán cổ điển,
  luôn chạy được, làm baseline so sánh.

Sự khác biệt forensic quan trọng (đây là lý do module tồn tại):
- synthforge DÁN PIXEL THÔ → twin descriptors giữ nguyên → copy-move
  detectors bắt được.
- modelforge dùng model SINH NỘI DUNG MỚI cho vùng bị xoá → không còn twin
  → keypoint/block đúng-luật-là-không-bắt-được; bằng chứng chỉ còn ở
  noise/freq/splicing cues. Đây là experiment L3-prep thật sự.

Các kiểu edit (kind):
- `erase`          : xoá object bằng inpainting — M = vùng bị xoá.
- `move`           : clone object sang chỗ mới (seamlessClone) RỒI inpaint
                     vùng gốc → object "đã dịch chuyển", source bị xoá →
                     KHÔNG còn twin. M = src ∪ dst.
- `splice_keep`    : seamlessClone sang chỗ mới, GIỮ nguyên bản gốc → twin
                     tồn tại nhưng mép đã blend — thử detector với twin "mềm".
- `splice_foreign` : seamlessClone vùng từ ảnh KHÁC vào — splicing đúng
                     nghĩa, không twin.

Output giữ contract synthforge: `*_O.png` (original), `*_F.png` (forged),
`*_M.png` (mask GT, 255 = vùng bị chỉnh), `meta.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from imageguard.evaluation.synthforge import _base_image

MODEL_PATH = Path("models/lama_fp32.onnx")
KINDS = ("erase", "move", "splice_keep", "splice_foreign")


class LaMaEditor:
    """Wrapper onnxruntime cho LaMa 512×512 fixed-input.

    I/O contract (theo model card, đã verify bằng inference thật):
      image: float32[1,3,512,512] RGB, /255
      mask : float32[1,1,512,512], 1 = xoá
      out  : float32[1,3,512,512] RGB 0-255
    Chiến lược: crop context quanh mask → resize 512 → inference → resize
    về → blend chỉ trong mask (feather) — giống cách Gumka/lama-cleaner
    chạy full-resolution.
    """

    INPUT = 512

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"không thấy {model_path} — tải theo docs/modelforge.md")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"])

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return bgr
        # context crop: bbox mask + pad 1 cạnh để model thấy xung quanh
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ph, pw = y1 - y0, x1 - x0
        pad = int(max(ph, pw) * 0.6) + 16
        cy0, cx0 = max(0, y0 - pad), max(0, x0 - pad)
        cy1, cx1 = min(h, y1 + pad), min(w, x1 + pad)
        crop = bgr[cy0:cy1, cx0:cx1]
        cmask = mask[cy0:cy1, cx0:cx1]

        s = self.INPUT
        img_in = cv2.resize(crop, (s, s)).astype(np.float32)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB) / 255.0
        img_in = img_in.transpose(2, 0, 1)[None]
        m_in = cv2.resize(cmask, (s, s),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        m_in = (m_in > 127).astype(np.float32)[None, None]

        out = self._sess.run(None, {"image": img_in, "mask": m_in})[0][0]
        out = out.transpose(1, 2, 0)
        out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        out = cv2.resize(out, (crop.shape[1], crop.shape[0]))

        res = bgr.copy()
        feather = cv2.GaussianBlur(
            (cmask > 0).astype(np.float32), (15, 15), 0)[..., None]
        res[cy0:cy1, cx0:cx1] = (
            crop * (1 - feather) + out * feather).astype(np.uint8)
        return res

    def clone(self, patch: np.ndarray, pmask: np.ndarray,
              dst: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        return cv2.seamlessClone(patch, dst, pmask, center, cv2.NORMAL_CLONE)


class ClassicEditor:
    """cv2.inpaint Telea + seamlessClone — không cần model file."""

    def inpaint(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)

    def clone(self, patch: np.ndarray, pmask: np.ndarray,
              dst: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        return cv2.seamlessClone(patch, dst, pmask, center, cv2.NORMAL_CLONE)


def make_editor(backend: str):
    if backend == "lama":
        return LaMaEditor()
    return ClassicEditor()


def _rand_region(rng: np.random.Generator, size: tuple[int, int],
                 lo=0.15, hi=0.28) -> tuple[int, int, int, int]:
    h, w = size
    rh = int(h * rng.uniform(lo, hi))
    rw = int(w * rng.uniform(lo, hi))
    y = int(rng.integers(4, h - rh - 4))
    x = int(rng.integers(4, w - rw - 4))
    return x, y, rw, rh


_SHAPES = ("rect", "ellipse", "polygon", "blob")


def _shape_mask(size: tuple[int, int], rect,
                shape: str = "ellipse",
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Mask GT theo 4 dạng — polygon/blob mô phỏng vùng chọn tay thật hơn
    ellipse suông (boundary không đều → seam khó hơn cho model)."""
    x, y, w, h = rect
    m = np.zeros(size, np.uint8)
    cx, cy = x + w // 2, y + h // 2
    if shape == "rect":
        cv2.rectangle(m, (x, y), (x + w, y + h), 255, -1)
    elif shape == "polygon" and rng is not None:
        n = int(rng.integers(5, 9))
        pts = [(int(cx + (w / 2) * rng.uniform(0.55, 1.0) * np.cos(2 * np.pi * i / n)),
                int(cy + (h / 2) * rng.uniform(0.55, 1.0) * np.sin(2 * np.pi * i / n)))
               for i in range(n)]
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    elif shape == "blob" and rng is not None:
        for _ in range(int(rng.integers(3, 6))):
            ex = int(cx + rng.integers(-w // 4, w // 4 + 1))
            ey = int(cy + rng.integers(-h // 4, h // 4 + 1))
            ew = max(8, int(w * rng.uniform(0.35, 0.6)))
            eh = max(8, int(h * rng.uniform(0.35, 0.6)))
            cv2.ellipse(m, (ex, ey), (ew // 2, eh // 2), 0, 0, 360, 255, -1)
    else:  # ellipse mặc định
        cv2.ellipse(m, (cx, cy), (w // 2, h // 2), 0, 0, 360, 255, -1)
    return m


def _ellipse_mask(size: tuple[int, int], rect) -> np.ndarray:
    return _shape_mask(size, rect, "ellipse")


def forge(
    base: np.ndarray,
    kind: str,
    editor,
    rng: np.random.Generator,
    foreign: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trả (edited, mask_gt, spec). mask=255 tại vùng bị chỉnh sửa."""
    h, w = base.shape[:2]
    out = base.copy()
    m_gt = np.zeros((h, w), np.uint8)
    spec: dict = {"kind": kind}

    shape = str(rng.choice(_SHAPES))
    spec["shape"] = shape

    if kind == "erase":
        rect = _rand_region(rng, (h, w))
        em = _shape_mask((h, w), rect, shape, rng)
        out = editor.inpaint(out, em)
        m_gt[em > 0] = 255
        spec["rect"] = list(rect)

    elif kind == "move":
        srect = _rand_region(rng, (h, w))
        drect = _rand_region(rng, (h, w))
        sx, sy, sw, sh = srect
        dx, dy, dw, dh = drect
        patch = base[sy:sy + sh, sx:sx + sw].copy()
        pm = np.full((sh, sw), 255, np.uint8)
        cv2.ellipse(pm, (sw // 2, sh // 2), (sw // 2 - 2, sh // 2 - 2),
                    0, 0, 360, 255, -1)
        out = editor.clone(
            patch, pm, out, (dx + dw // 2, dy + dh // 2))
        sm = _shape_mask((h, w), srect, shape, rng)
        out = editor.inpaint(out, sm)  # xoá gốc — object đã "đi"
        m_gt[sm > 0] = 255
        dm = _ellipse_mask((h, w), drect)
        m_gt[dm > 0] = 255
        spec.update(src=list(srect), dst=list(drect))

    elif kind in ("splice_keep", "splice_foreign"):
        if kind == "splice_foreign" and foreign is not None:
            fh, fw = foreign.shape[:2]
            rect = _rand_region(rng, (fh, fw))
            fx, fy, fw2, fh2 = rect
            patch = foreign[fy:fy + fh2, fx:fx + fw2].copy()
        else:
            rect = _rand_region(rng, (h, w))
            fx, fy, fw2, fh2 = rect
            patch = base[fy:fy + fh2, fx:fx + fw2].copy()
        pm = np.full((fh2, fw2), 255, np.uint8)
        cv2.ellipse(pm, (fw2 // 2, fh2 // 2), (fw2 // 2 - 2, fh2 // 2 - 2),
                    0, 0, 360, 255, -1)
        while True:
            dx = int(rng.integers(4, w - fw2 - 4))
            dy = int(rng.integers(4, h - fh2 - 4))
            if kind == "splice_keep" and \
                    abs(dx - fx) < fw2 and abs(dy - fy) < fh2:
                continue  # dest trùng source → vô nghĩa, chọn lại
            break
        out = editor.clone(patch, pm, out, (dx + fw2 // 2, dy + fh2 // 2))
        dm = np.zeros((h, w), np.uint8)
        cv2.ellipse(dm, (dx + fw2 // 2, dy + fh2 // 2),
                    (fw2 // 2, fh2 // 2), 0, 0, 360, 255, -1)
        m_gt[dm > 0] = 255
        spec.update(src=list(rect), dst=[dx, dy, fw2, fh2],
                    foreign=kind == "splice_foreign")

    else:
        raise ValueError(f"kind không hỗ trợ: {kind}")
    return out, m_gt, spec


def generate_set(
    n: int,
    seed: int,
    out_dir: Path,
    size: tuple[int, int] = (512, 512),
    backend: str = "lama",
    kinds: tuple[str, ...] = KINDS,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    editor = make_editor(backend)
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        base = _base_image(rng, size)
        foreign = _base_image(rng, size) if kind == "splice_foreign" else None
        forged, mask, spec = forge(base, kind, editor, rng, foreign)
        stem = f"{kind}_{i:04d}"
        # Một phần lưu JPEG q90 — forged được re-save thực tế + cho ELA có
        # tín hiệu (PNG → ELA luôn NOT_APPLICABLE làm suy giảm ma trận).
        save_jpeg = bool(rng.integers(0, 2))
        ext, param = (".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]) if save_jpeg \
            else (".png", [])
        cv2.imwrite(str(out_dir / f"{stem}_O.png"), base)
        cv2.imwrite(str(out_dir / f"{stem}_F{ext}"), forged, param)
        cv2.imwrite(str(out_dir / f"{stem}_M.png"), mask)
        entries.append({"id": stem, "seed": seed, "kind": kind,
                        "backend": backend, "spec": spec,
                        "format": "jpeg" if save_jpeg else "png"})
    manifest = out_dir / "meta.json"
    manifest.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(prog="modelforge")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--backend", choices=["lama", "classic"], default="lama")
    a = ap.parse_args()
    mf = generate_set(a.n, a.seed, a.out, (a.size, a.size), a.backend)
    print(f"manifest: {mf}  backend={a.backend}")


if __name__ == "__main__":
    main()
