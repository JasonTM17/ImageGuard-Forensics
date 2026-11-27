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

## 2. CoC Xception — screen-recapture deepfake (`deepfake_screen`)

| Mục | Giá trị |
| --- | --- |
| File | `models/coc_xception_screen_deepfake.onnx` |
| Kích thước | 83 165 816 bytes (~79 MB) |
| SHA-256 | `800dff109c5bbb165b9b0404e1e1207bc521ec75fc51926247cbae2abe271b1e` |
| Nguồn checkpoint | Zenodo record `23150212`, file `ScreenDeepfakeBench_models.zip`, member `checkpoints/coc.pth` |
| Checkpoint SHA-256 | `47889d19bcb55b4c3ba59c658f046e7dac4f8a018ef1148e47ea40937a20815e` |
| Kiến trúc/dữ liệu | Xception train bởi tác giả Caught on Camera cho deepfake phát qua màn hình bị camera quay lại |
| Runtime | `onnxruntime` CPU; không dùng PyTorch khi phân tích |

Checkpoint `.pth` không chạy trực tiếp trong ImageGuard. Để tái tạo ONNX,
clone `SENSE-Lab-Security/ScreenDeepfakeBench`, tải `ScreenDeepfakeBench_models.zip`
từ `https://zenodo.org/records/23150212`, rồi trong môi trường Python 3.10 có torch 2.2.x chạy:

```bash
python scripts/export_coc_xception_onnx.py \n  --repo-root path/to/ScreenDeepfakeBench \n  --checkpoint path/to/checkpoints/coc.pth \n  --output models/coc_xception_screen_deepfake.onnx
```

Pipeline runtime dùng YuNet chỉ để định vị mặt/landmark, sau đó similarity-align
5 điểm theo preprocessing DeepfakeBench/CoC trước khi chấm điểm. Bài báo tác
giả công bố frame ACC 89,46% và frame AUC 0,9654 trên benchmark của họ; đây là
số liệu của tác giả, không phải hiệu chuẩn độc lập của ImageGuard.

Licence/trọng số: repo ScreenDeepfakeBench kế thừa DeepfakeBench **CC BY-NC 4.0**
và các điều khoản nguồn dữ liệu liên quan. Chỉ dùng cho mục đích phi thương mại,
nghiên cứu hoặc học thuật; ghi công Caught on Camera/ScreenDeepfakeBench khi
báo cáo kết quả.

## 3. YuNet — face detection (phục vụ `deepfake_ml` và `deepfake_screen`)

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

## 4. NAFNet — deblurring (đã có từ trước)

Xem `docs/packaging.md` — pin `07263f41…` trong `enhancement/nafnet.py`.
