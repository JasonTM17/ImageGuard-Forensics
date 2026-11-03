"""synthforge — synthetic copy-move forgery generator, deterministic theo seed.

Tại sao generator nội bộ từ ngày 1: detector cần ground-truth mask để test —
chờ dataset ngoài = không test được gì. Ảnh procedural (gradient + noise +
shapes) đảm bảo CÓ texture để keypoint/DCT hoạt động; thiếu texture detector
im lặng hợp lệ (INCONCLUSIVE), không phải bug.

Mỗi mẫu ghi: base_O.png, forged_F.png, mask_M.png (255=vùng forged),
meta.json{seed,src_polygon,dst_offset,transform,postproc,tier}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TIERS = ("easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class ForgeSpec:
    """Đặc tả một vụ giả mạo — rng sinh ra, ghi vào meta.json để tái hiện."""

    size: tuple[int, int]
    shape: str  # rect | ellipse | polygon
    src_polygon: list[list[int]]
    dst_offset: tuple[int, int]
    rotate_deg: float  # 0 = translate thuần
    scale: float
    postproc: str  # none | jpeg80 | jpeg60 | blur | noise
    tier: str


def _base_image(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Ảnh nền có texture đủ cho SIFT/DCT: gradient + shapes + AWGN."""
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    img = np.tile(x, (h, 1)) * 0.5 + np.tile(y, (w, 1)).T * 0.5
    img = cv2.merge([img, np.roll(img, 32, axis=1), np.roll(img, 16, axis=0)])
    # Vài hình ngẫu nhiên để tạo cạnh/góc — keypoint cần góc, không phải gradient.
    for _ in range(rng.integers(6, 14)):
        c = tuple(int(v) for v in rng.integers(30, 225, 3))
        if rng.random() < 0.5:
            cv2.rectangle(
                img,
                tuple(rng.integers(0, w - 60, 2).tolist()),
                tuple(rng.integers(60, w, 2).tolist()),
                c,
                -1,
            )
        else:
            cv2.circle(
                img,
                (int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))),
                int(rng.integers(15, 60)),
                c,
                -1,
            )
    img += rng.normal(0, 4, img.shape).astype(np.float32)  # AWGN nhẹ — thực tế
    return np.clip(img, 0, 255).astype(np.uint8)


def _rand_spec(rng: np.random.Generator, tier: str, size: tuple[int, int]) -> ForgeSpec:
    w, h = size
    shape = str(rng.choice(["rect", "ellipse", "polygon"]))
    # Vùng copy 8–20% cạnh — đủ lớn để detector thấy, đủ nhỏ để thực tế.
    rw, rh = int(w * rng.uniform(0.12, 0.25)), int(h * rng.uniform(0.12, 0.25))
    sx, sy = int(rng.integers(0, w - rw)), int(rng.integers(0, h - rh))
    dx = int(rng.integers(-w // 3, w // 3)) or 10
    dy = int(rng.integers(-h // 3, h // 3)) or 10
    # Đảm bảo dst nằm trong ảnh và không trùng src.
    dx = int(np.clip(sx + dx, 0, w - rw) - sx) or rw // 2
    dy = int(np.clip(sy + dy, 0, h - rh) - sy) or rh // 2
    src_poly = [[sx, sy], [sx + rw, sy], [sx + rw, sy + rh], [sx, sy + rh]]
    rotate = float(rng.uniform(-25, 25)) if tier in ("medium", "hard") else 0.0
    scale = float(rng.uniform(0.85, 1.15)) if tier == "hard" else 1.0
    postproc = str(
        rng.choice(
            ["none", "jpeg80"]
            if tier == "easy"
            else ["none", "jpeg80", "jpeg60", "blur", "noise"]
        )
    )
    return ForgeSpec(size, shape, src_poly, (dx, dy), rotate, scale, postproc, tier)


def forge(base: np.ndarray, spec: ForgeSpec) -> tuple[np.ndarray, np.ndarray]:
    """Áp một vụ copy-move lên base. Trả (forged_bgr, mask_uint8 0/255)."""
    h, w = base.shape[:2]
    forged = base.copy()
    poly = np.array(spec.src_polygon, dtype=np.int32)
    x, y, rw, rh = cv2.boundingRect(poly)

    patch = base[y : y + rh, x : x + rw].copy()
    mask_patch = np.zeros((rh, rw), np.uint8)
    rel = poly - [x, y]
    if spec.shape == "ellipse":
        cv2.ellipse(mask_patch, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
    elif spec.shape == "polygon":  # giữ nguyên polygon bao rect — tier khó hơn
        cv2.fillPoly(mask_patch, [rel], 255)
    else:
        mask_patch[:] = 255

    # Transform: rotate+scale quanh tâm patch → affine 2×3.
    M = cv2.getRotationMatrix2D((rw / 2, rh / 2), spec.rotate_deg, spec.scale)
    patch = cv2.warpAffine(patch, M, (rw, rh), flags=cv2.INTER_LINEAR)
    mask_patch = cv2.warpAffine(mask_patch, M, (rw, rh), flags=cv2.INTER_NEAREST)

    dx, dy = spec.dst_offset
    region = forged[y + dy : y + dy + rh, x + dx : x + dx + rw]
    if region.shape[:2] != (rh, rw):
        return forged, np.zeros((h, w), np.uint8)  # dst lòi khỏi ảnh — skip
    a = (mask_patch > 127)[:, :, None]
    forged[y + dy : y + dy + rh, x + dx : x + dx + rw] = np.where(a, patch, region)

    mask = np.zeros((h, w), np.uint8)
    mask[y + dy : y + dy + rh, x + dx : x + dx + rw] = mask_patch
    return forged, mask


def _postproc(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "jpeg80":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)
    if kind == "jpeg60":
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)
    if kind == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if kind == "noise":
        return np.clip(img.astype(np.float32) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    return img


def generate_set(
    n: int, seed: int, out_dir: Path, size: tuple[int, int] = (512, 512), tier: str = "medium"
) -> Path:
    """Sinh n mẫu vào out_dir. Deterministic: cùng seed → manifest byte-identical.
    Trả path manifest json."""
    assert tier in TIERS
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    entries = []
    for i in range(n):
        base = _base_image(rng, size)
        spec = _rand_spec(rng, tier, size)
        forged, mask = forge(base, spec)
        forged = _postproc(forged, spec.postproc, rng)
        stem = f"{tier}_{i:04d}"
        cv2.imwrite(str(out_dir / f"{stem}_O.png"), base)
        cv2.imwrite(str(out_dir / f"{stem}_F.png"), forged)
        cv2.imwrite(str(out_dir / f"{stem}_M.png"), mask)
        entries.append({"id": stem, "seed": seed, "spec": _spec_dict(spec)})
    manifest = out_dir / "meta.json"
    manifest.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return manifest


def _spec_dict(s: ForgeSpec) -> dict:
    return {
        "shape": s.shape,
        "src_polygon": s.src_polygon,
        "dst_offset": list(s.dst_offset),
        "rotate_deg": round(s.rotate_deg, 3),
        "scale": round(s.scale, 4),
        "postproc": s.postproc,
        "tier": s.tier,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="synthforge")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--tier", choices=TIERS, default="medium")
    args = ap.parse_args(argv)
    m = generate_set(args.n, args.seed, args.out, tier=args.tier)
    print(f"Wrote {args.n} samples to {args.out} (manifest: {m})")


if __name__ == "__main__":
    main()
