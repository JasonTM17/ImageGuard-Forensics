# Model bên thứ ba (tùy chọn, pin SHA-256)

ImageGuard không bundle weights vào repo (`models/` đã gitignore). Mỗi model
là tùy chọn, tải thủ công, và method tương ứng tự trả `NOT_APPLICABLE` khi
thiếu file — không bao giờ tải lúc runtime, không gọi network lúc phân tích.

## 1. GenConViT-ED — deepfake face detection (`deepfake_ml`)

| Mục | Giá trị |
| --- | --- |
| File | `models/genconvit_ed_inference.onnx` |
| Kích thước | 122 433 215 bytes (~117 MB) |
| SHA-256 | `59605cda6b03322d1c2d892b4ba058d0a004b6c79b886fdbc2f420c897f7b9ae` |
| Nguồn ONNX | `huggingface.co/pranjal-pravesh/GenConViT-onnx` (revision `b1078f3c`) |
| Model gốc | `erprogs/GenConViT` — Wodajo et al., *Applied Sciences* 2025, train trên DFDC / FaceForensics++ / Celeb-DF v2 / DeepfakeTIMIT |
| Runtime | `onnxruntime` (đã có nếu cài extra `lama`) |

Tải (PowerShell/bash đều được):

```bash
curl -L -o models/genconvit_ed_inference.onnx \
  "https://huggingface.co/pranjal-pravesh/GenConViT-onnx/resolve/b1078f3c9c92dab1df2cdf019e149f43770d9160/genconvit_ed_inference.onnx"
```

Kiểm tra hash sau tải — method từ chối chạy (verdict `ERROR`) nếu size/hash
không khớp pin.

### Licence — đọc trước khi dùng

- Code GenConViT gốc: **MIT** (repo `erprogs/GenConViT`).
- **Weights gốc** trên HuggingFace (`Deressa/GenConViT`): model card ghi
  **CC BY-NC 4.0 — chỉ dùng phi thương mại**. Bản ONNX convert ghi MIT nhưng
  bản convert không thể thay đổi licence của weights gốc; ImageGuard vì vậy
  áp dụng đọc hiểu bảo thủ: **chức năng này chỉ dành cho mục đích phi
  thương mại** (nghiên cứu, học thuật, capstone, điều tra nội bộ không
  thương mại), kèm ghi công tác giả khi xuất báo cáo có dùng kết quả.
- Ghi công: Wodajo, D. et al. "Deepfake Video Detection Using Generative
  Convolutional Vision Transformer", *Applied Sciences* 15(12):6622, 2025.

## 2. YuNet — face detection (phục vụ `deepfake_ml`)

| Mục | Giá trị |
| --- | --- |
| File | `models/face_detection_yunet_2023mar.onnx` |
| Kích thước | 232 589 bytes (~232 KB) |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Nguồn | `github.com/opencv/opencv_zoo` — `models/face_detection_yunet` (MIT) |

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

YuNet chỉ *định vị* khuôn mặt để crop đưa vào classifier — không nhận diện
danh tính, không lưu embedding.

## 3. NAFNet — deblurring (đã có từ trước)

Xem `docs/packaging.md` — pin `07263f41…` trong `enhancement/nafnet.py`.
