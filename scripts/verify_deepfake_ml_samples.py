"""Numeric-only deepfake-ML verification on the GenConViT sample cohort.

Đánh giá method ``deepfake_ml`` trên 12 video mẫu từ repo erprogs/GenConViT
(MIT) đặt tại ``data/external/genconvit-samples/`` — 2 video fake
(0017_fake, 0048_fake) và 10 video real theo tên file. Đây là cohort
verification nhỏ độc lập với synthetic fixtures: nó chứng minh pipeline
end-to-end phân biệt được real/fake trên dữ liệu thật, nhưng KHÔNG phải
benchmark độ chính xác: 12 video không có nhãn ground-truth đáng tin (tên file
không chứa nhãn trong DFDC/FF++), nguồn trùng với nhà phát triển model.
Giá trị duy nhất: smoke pipeline end-to-end trên dữ liệu deepfake thật.

In JSON duy nhất ra stdout: per-video aggregation + phân tích tách ngưỡng.
Không ghi file, không copy ảnh, không ghi DB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imageguard.core.config import load_thresholds
from imageguard.forensics.ml.deepfake_genconvit import (
    aggregate_fake_frac,
    infer_sigmoids,
    preprocess_crop,
    verdict_for_score,
)
from imageguard.forensics.ml.face_detect import crop_face, detect_primary_face

FRAMES_PER_VIDEO = 10


def analyze_video(path: Path, detect_max: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, min(FRAMES_PER_VIDEO, total)).astype(int)
    crops = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        detection = detect_primary_face(frame, detect_max)
        if detection is None:
            continue
        rect, _conf = detection
        crops.append(crop_face(frame, rect, 224))
    cap.release()
    if not crops:
        return {"video": path.name, "faces": 0}

    tensor = np.concatenate([preprocess_crop(c) for c in crops], axis=0)
    sigmoids = infer_sigmoids(tensor)
    frac, sig_fake, sig_real = aggregate_fake_frac(sigmoids)
    per_frame = sorted(
        round(float(s[0] / (s[0] + s[1])), 4) for s in sigmoids if s[0] + s[1] > 0
    )
    return {
        "video": path.name,
        "faces": len(crops),
        "fake_frac": round(frac, 4),
        "mean_sig_fake": round(sig_fake, 4),
        "mean_sig_real": round(sig_real, 4),
        "frame_frac_min": per_frame[0],
        "frame_frac_median": per_frame[len(per_frame) // 2],
        "frame_frac_max": per_frame[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/genconvit-samples")
    )
    args = parser.parse_args()

    cfg = load_thresholds()
    fake_t = float(cfg.value("deepfake_ml", "fake_thresh"))
    real_t = float(cfg.value("deepfake_ml", "real_thresh"))
    detect_max = int(cfg.value("deepfake_ml", "detect_max_side"))

    rows = []
    t0 = time.perf_counter()
    for vp in sorted(args.root.iterdir()):
        if not vp.is_file():
            continue
        row = analyze_video(vp, detect_max)
        # KHÔNG phải ground truth: tên file không mang nhãn trong các dataset
        # deepfake (DFDC đặt tên hash cho cả real lẫn fake; sample pack của repo
        # tác giả chủ yếu là video fake dùng demo). Chỉ là hint để đọc kết quả.
        row["filename_hint"] = "FAKE" if "fake" in vp.name.lower() else "unknown"
        if row.get("fake_frac") is not None:
            row["verdict"] = verdict_for_score(row["fake_frac"], fake_t, real_t).value
        rows.append(row)

    scored = [r for r in rows if r.get("fake_frac") is not None]
    fakes = [r for r in scored if r["filename_hint"] == "FAKE"]
    reals = [r for r in scored if r["filename_hint"] != "FAKE"]
    min_fake = min((r["fake_frac"] for r in fakes), default=None)
    max_real = max((r["fake_frac"] for r in reals), default=None)
    # Per-frame worst case: frame thật bị đẩy cao nhất vs frame fake thấp nhất
    min_fake_frame = min((r["frame_frac_min"] for r in fakes), default=None)
    max_real_frame = max((r["frame_frac_max"] for r in reals), default=None)
    separated = (
        min_fake is not None and max_real is not None and min_fake > max_real
    )
    report = {
        "cohort": "genconvit-sample-prediction-data (12 videos, MIT)",
        "videos": len(rows),
        "faces_total": sum(r.get("faces", 0) for r in rows),
        "thresholds": {"fake": fake_t, "real": real_t},
        "min_fake_frac_aggregated": min_fake,
        "max_real_frac_aggregated": max_real,
        "min_fake_frac_per_frame": min_fake_frame,
        "max_real_frac_per_frame": max_real_frame,
        "separation_clean_aggregated": separated,
        "warning": "filename_hint KHÔNG phải ground truth — cohort này không validate accuracy; chỉ smoke pipeline end-to-end",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if separated else 1


if __name__ == "__main__":
    raise SystemExit(main())
