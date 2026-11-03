# ML Extension — Level 3 Design (design-only, chưa implement)

## Vì sao để design-only

Level 1–2 (traditional DIP) đã đủ cho capstone + học thuật toán nền.
ML/DL cần dataset lớn + GPU + framework nặng (torch ~2GB) — scope creep rõ
rệt nếu nhét vào MVP. Tài liệu này là spec để implement sau mà không phá
kiến trúc hiện tại.

## Điều kiện tiên quyết

- `torch` wheel cho Python hiện tại: `pip index versions torch` — yêu cầu
  cp314 wheel trước khi commit dep; nếu chưa có → pin python 3.12 venv riêng
  cho nhánh ML.
- Dataset: CoMoFoD (train) + CASIA v2/IMD2020 (validation), giữ nguyên
  quy tắc split-by-base-image của `evaluation/`.
- Giữ `ForensicReport`/`AlgorithmResult` contract — model chỉ là một
  `Method` mới trong registry, KHÔNG phá fusion/five-state verdict.

## Kiến trúc đề xuất: U-Net localization baseline

- Input: RGB 256×256 patches (stride 128) + kênh phụ tùy chọn (ELA heat,
  noise residual) — ablation ghi vào report.
- Output: mask prob per pixel → threshold 0.5 → mask → `AlgorithmResult`
  với `score_definition = "mean max-prob trên vùng mask"`.
- Loss: BCE + soft-IoU (imbalance: vùng giả nhỏ); augment copy-move từ
  `synthforge` (đã deterministic).
- Verdict map: `SUSPICIOUS_INDICATION` chỉ khi coverage ≥ ngưỡng và
  calibrated trên dev split — vẫn đi qua fusion rule chung; model cũng có
  thể trả INCONCLUSIVE khi OOD (score thấp toàn cục).
- Cross-dataset eval bắt buộc: train CoMoFoD → test CASIA → report
  generalization gap (traditional methods không cần train nên không có
  gap này — đây là điểm so sánh chính trong báo cáo capstone).

## So sánh traditional vs DL (bảng cho báo cáo)

| Tiêu chí | Keypoint/Block (hiện tại) | U-Net (đề xuất) |
|---|---|---|
| Cần dữ liệu train | Không | Có (hàng nghìn ảnh) |
| Giải thích được | Cao (cặp match, shift vote) | Thấp (heatmap prob) |
| Distortion mạnh (blur/rotate nặng) | Yếu | Tốt hơn nếu augment đủ |
| Reproduce | deterministic, seed-locked | cần checkpoint + seed |
| Runtime CPU | giây/ảnh | nhanh hơn nếu có GPU |

## Không làm

- Không dùng API ngoài (vi phạm mục tiêu học DIP + offline).
- Không claim "độ chính xác X%" — mọi số đo qua `evaluation/` hiện có.
