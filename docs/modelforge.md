# ModelForge + ModelBench

## Vai trò

LaMa ONNX là model inpainting dùng để **tạo ảnh thử nghiệm**. Desktop app
không dùng LaMa để phát hiện hay kết luận giả mạo. App phân tích bằng SIFT/ORB, block DCT và
các heuristic forensic; U-Net vẫn chỉ là thiết kế trong `ml-extension.md`.

ModelBench đo detector trên ảnh synthetic có ground truth, gồm IoU và
boundary-F1@8. ModelBench cũng chạy từng ảnh gốc chưa chỉnh sửa như negative
control để ghi verdict, tỷ lệ mask không rỗng và diện tích mask. Các control
này vẫn là ảnh synthetic, không đại diện ảnh chụp thật.

## Các kiểu chỉnh sửa

| kind | Thay đổi | Kỳ vọng giới hạn |
|---|---|---|
| `erase` | LaMa lấp vùng bị xóa | Không còn bản sao pixel để keypoint/block đối chiếu |
| `move` | Clone patch sang vị trí khác rồi lấp vùng nguồn | Có thể còn dấu vết; source và destination không chồng nhau |
| `splice_keep` | Clone vùng nhưng giữ nguồn | Bản sao còn lại, keypoint có thể bắt được |
| `splice_foreign` | Clone vùng từ ảnh khác | Không phải copy-move; cue noise/splicing chỉ mang tính khám phá |

Mỗi bộ có mask GT, manifest, seed và backend. Các bộ mới ghi thêm SHA-256 của
model LaMa. `move` destination dùng cùng kích thước với patch thật, đặt GT bằng
đúng clone mask tại vị trí clone; source và destination không chồng nhau.

## Kết quả sau khi sửa ground truth

Các số dưới đây đo trên bộ mới, dùng ngưỡng hiện có mà không hiệu chỉnh lại. Ảnh benchmark được sinh trong thư mục `<temp-dir>` vì các bộ dataset trong repo không được phép ghi đè:

| Bộ | N | Seed | Kích thước | Model | config_hash | dataset_sha256 | git_sha |
|---|---:|---:|---:|---|---|---|---|
| Train | 12 | 2026 | 384×384 | LaMa ONNX SHA-256 `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` | `df80dcb9a6d32070` | `46b367bc6fc6623edcc4d4a4eb790ca23896fceea17655511a0165835e46a7ea` | (bench JSON, xem dưới) |
| Holdout | 6 | 7777 | 512×512 | cùng SHA-256 | `df80dcb9a6d32070` | `8198a6e80fe2135cf83922afc13fbc7f2a1fc84cfb0f869eb7cd779602fbe33d` | (bench JSON, xem dưới) |

Lệnh sinh và bench (chạy với Python trong `.venv\Scripts\python.exe`):

```powershell
# Train
$py = .\.venv\Scripts\python.exe
$train = Join-Path $env:TEMP 'ig-phaseb-train'
& $py -m imageguard.evaluation.modelforge --n 12 --seed 2026 --out $train --backend lama --size 384
& $py -m imageguard.evaluation.modelbench --set $train --methods keypoint,block,noise,noise_local,freq,splicing,ela --out (Join-Path $env:TEMP 'ig-phaseb-bench-train')

# Holdout
$hold = Join-Path $env:TEMP 'ig-phaseb-holdout'
& $py -m imageguard.evaluation.modelforge --n 6 --seed 7777 --out $hold --backend lama --size 512
& $py -m imageguard.evaluation.modelbench --set $hold --methods keypoint,block,noise,noise_local,freq,splicing,ela --out (Join-Path $env:TEMP 'ig-phaseb-bench-holdout')
```

IoU/bF1 là trung bình trên mọi mẫu có GT non-empty và mask dự đoán đúng hình
học; mask dự đoán thiếu được tính bằng 0, còn mask sai hình học bị loại khỏi
trung bình và đếm riêng ở `n_malformed_gt`. `INCONCLUSIVE` vẫn là kết quả thận
trọng, không phải xác nhận ảnh có chỉnh sửa.

| Bộ / loại | Keypoint verdict; IoU; bF1 | `noise_local` verdict; IoU; bF1 |
|---|---|---|
| Train erase (n=3) | 3 inconclusive; 0.000; 0.016 | 2 inconclusive; 0.335; 0.559 |
| Train move (n=3) | 3 inconclusive; 0.000; 0.005 | 3 inconclusive; 0.368; 0.671 |
| Train splice_keep (n=3) | 3 suspicious; 0.319; 0.598 | 3 no indication; 0.000; 0.000 |
| Train splice_foreign (n=3) | 3 inconclusive; 0.001; 0.016 | 3 no indication; 0.000; 0.000 |
| Holdout erase (n=2) | 2 inconclusive; 0.000; 0.005 | 2 inconclusive; 0.665; 0.888 |
| Holdout move (n=2) | 2 suspicious; 0.042; 0.293 | 2 inconclusive; 0.376; 0.655 |
| Holdout splice_keep (n=1) | 1 suspicious; 0.319; 0.595 | 1 inconclusive; 0.000; 0.000 |
| Holdout splice_foreign (n=1) | 1 no indication; 0.000; 0.000 | 1 no indication; 0.000; 0.000 |

The keypoint `move` holdout rows show why verdict and localization must be read
together: both returned `SUSPICIOUS_INDICATION`, but mean IoU was only 0.042. The
`noise_local` method never returns `SUSPICIOUS_INDICATION`; it supplies a
smooth-region cue and can return `INCONCLUSIVE`.

On untouched synthetic originals, `noise_local` emitted a non-empty mask on
1/12 train images (mean area 0.101%, maximum 1.215%) and 1/6 holdout images
(mean area 0.098%, maximum 0.586%). Both controls had one `INCONCLUSIVE` and no
`SUSPICIOUS_INDICATION` verdict. Keypoint emitted candidate masks on all 18
originals but returned no `SUSPICIOUS_INDICATION` verdict. This is not a
real-photo false-positive rate.

Block DCT had mean IoU 0 across all four edit types in these LaMa sets. It is
not a suitable detector for edits that replace pixels instead of preserving a
copy-move twin. Keypoint marked all `splice_keep` samples
`SUSPICIOUS_INDICATION` here with mean IoU 0.319 (train n=3) and 0.319 (holdout
n=1); high verdict rate with moderate localization — đọc cùng IoU/bF1, không chỉ
verdict. Sample counts remain too small for general claims.

**Traceability (re-run 2026-11-12):** `python -m imageguard.evaluation.modelbench`
on `data/modelforge_audit_train_fixed` và `data/modelforge_audit_holdout_fixed`,
output ngoài repo. Bench JSON SHA-256: train
`e70d58546d87e515c437563ff0c6b6253d33e47a4669ec2e7a3f7e712c6855ed`, holdout
`33c56479c9743eb020a35144ae89a139c50aa802d942dc36ce62b40707354326`. Khớp báo
cáo audit markdown cục bộ (gitignored, không commit). Các số splice_keep IoU/bF1
cũ (0.006/0.139 train; 0.001/0.066 holdout) là sai so với artifact này.

## Legacy evidence and limits

An independent audit found that old `move` manifests sampled source and
destination rectangles independently. The cloned patch used source dimensions,
but its destination GT used unrelated destination dimensions. Old `move` IoU,
boundary-F1, and noise separation are therefore invalid. Older ModelBench
versions also averaged localization only over returned masks; those conditional
means are not comparable to the all-sample metrics above. The old
`modelbench_v4_*` artifacts lack dataset/source/config provenance and must not
be used as reproducible results.

The corrected sets are small and synthetic. CoMoFoD and test sets outside
this benchmark (such as root `modelforge_test`) were not present or used for this
audit. The values `0.6` and `0.45` remain provisional heuristics; they were not
retuned on the corrected sets and are not calibrated for real photographs. Metric
definitions and empty-mask rules are in [`evaluation-methodology.md`](evaluation-methodology.md).

## Reproduce

Markdown audit exports (`reports/audit-261104-fixed-train.md`,
`reports/audit-261104-fixed-holdout.md`) are gitignored — không nằm trong git.
Tái lập bằng modelbench trên các bộ `data/modelforge_audit_*_fixed` đã có sẵn
(không cần sinh lại ảnh; `data/` gitignored nên bản clone sạch không có các bộ
này):

```powershell
$py = .\.venv\Scripts\python.exe
$trainBench = Join-Path $env:TEMP 'ig-modelbench-train'
$holdBench = Join-Path $env:TEMP 'ig-modelbench-holdout'
& $py -m imageguard.evaluation.modelbench --set data/modelforge_audit_train_fixed `
  --methods keypoint,block,noise,noise_local,freq,splicing,ela `
  --out $trainBench
& $py -m imageguard.evaluation.modelbench --set data/modelforge_audit_holdout_fixed `
  --methods keypoint,block,noise,noise_local,freq,splicing,ela `
  --out $holdBench
```

Để sinh lại ảnh từ đầu (ghi đè local dataset gitignored), thêm các lệnh
`modelforge` với `--out` trỏ vào `data/modelforge_audit_*_fixed` như trên.
Output bench: cặp `.md` + `.json` tại `$trainBench` / `$holdBench`.

LaMa weights are local and are not committed. The current weight file's SHA-256
is `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`.

## Attribution

LaMa: Suvorov et al., WACV 2022, arXiv:2109.07161, Apache-2.0.
ONNX export: Carve/LaMa-ONNX (Apache-2.0), trained on Places2 (CC-BY 4.0).
