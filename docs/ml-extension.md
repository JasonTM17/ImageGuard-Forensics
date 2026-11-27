# ML Extension — mô hình DIP và thiết kế mạng sâu

## Nhánh đã triển khai: logistic trên đặc trưng DIP

Từ 2026-11-27, `deepfake_model.py`, `deepfake_training.py` và
`learned_deepfake.py` hỗ trợ học logistic regression bằng NumPy trên 13 đặc
trưng vùng mặt. Không cần GPU/framework deep learning. Nhóm tự cung cấp dữ
liệu có nhãn, fit dev, kiểm tra chéo theo nhóm nguồn, rồi đánh giá test độc
lập trước khi nạp JSON vào GUI/CLI. Không có trọng số hoặc accuracy thực tế
đi kèm. Xem [hướng dẫn đầy đủ](deepfake-evaluation.md).

## Nhánh đã triển khai: GenConViT-ED (ONNX)

Phương pháp tùy chọn `deepfake_ml` chạy trên crop mặt từ YuNet, độc lập với bộ phân loại DIP. Xem [cài đặt và giấy phép](third-party-models.md) và [giới hạn](limitations.md#deepfake-ml-deepfake_ml-genconvit-ed).

Phần U-Net định vị copy-move dưới đây vẫn là thiết kế chưa triển khai.

## Vì sao để design-only

Level 1–2 (traditional DIP) đã đủ cho capstone + học thuật toán nền.
Nhánh mạng sâu cần dataset lớn và thường dùng GPU/framework nặng (torch, ước lượng ~2 GB — chưa đo
trong repo) — scope creep rõ rệt nếu nhét vào MVP. Tài liệu này là spec để
implement sau mà không phá kiến trúc hiện tại.

## Điều kiện tiên quyết

- `torch` wheel cho Python hiện tại: `pip index versions torch` — yêu cầu
  cp314 wheel trước khi commit dep; nếu chưa có → pin Python 3.13 venv riêng
  cho nhánh ML (pyproject yêu cầu `>=3.13`).
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

## Hướng mở rộng: Thu nhận từ Camera (Webcam máy tính & Camera ngoài) + Phát hiện Deepfake

- **Nguồn camera đa dạng:** Kết hợp hai luồng đầu vào song song: (1) camera ngoài (IoT / IP Camera) qua `ingest/http_pull.py` và `ingest/camera.py` (`capture_iot_http_frame`), và (2) webcam tích hợp / USB camera ngoài trên máy tính (`cv2.VideoCapture`, chế độ chụp song song 2 camera) qua `CameraCaptureDialog` chạy ở luồng nền (`QThread`), trích khung hình tĩnh đưa vào `ImageDocument`.
- **Pipeline phát hiện Deepfake cổ điển đã tích hợp (`forensics/auxiliary/deepfake_cues.py`):**
  1. **Định vị ROI khuôn mặt 3 tầng:** Haar Cascade (nếu có) $\rightarrow$ màu da `YCrCb + HSV` + hình thái học + kiểm chứng hai mắt/SAT (`skin_haar_dip`) $\rightarrow$ vùng trung tâm chỉ để minh họa khi không có mặt; không cấp score ở fallback trung tâm.
  2. **4 trụ cột trích đặc trưng Xử lý ảnh thuần NumPy + OpenCV:** (i) nhiễu phần dư cảm biến `σ_in` vs `σ_out` & vòng biên `seam_ring`, (ii) vi kết cấu LBP 8 lân cận vector hóa & khoảng cách $\chi^2$, (iii) đỉnh tuần hoàn + suy hao hướng tâm phổ 2D FFT, (iv) hợp nhất thành điểm khám phá và `heatmap` đa lớp theo contract `AUX_METHODS`.
  3. **Hai nhánh học độc lập:** logistic NumPy dùng đặc trưng DIP; GenConViT-ED nhận crop mặt. Phân tích dùng ảnh gốc; super-resolution chỉ dùng để hiển thị.
