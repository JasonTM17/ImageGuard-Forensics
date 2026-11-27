# Kiểm tra độ ổn định cloud — 2026-11-27

## Phạm vi và môi trường

Đợt này kiểm tra thực thi, xử lý đầu vào lỗi, lưu trữ, vòng đời tác vụ GUI và
đóng gói. Nhãn và khuôn mặt tổng hợp trong probe chỉ phục vụ kiểm thử phần mềm;
không dùng chúng để công bố accuracy, AUC hay khả năng nhận diện Deepfake thực tế.

- Nền mã: `7866ebf143a888c31d5a82643f877f5815862d5a`, cộng các thay đổi trong đợt này.
- Máy chạy: Linux x86_64, glibc 2.41; Python 3.13.5, OpenCV 4.14.0, NumPy 2.5.5.
- GUI tự động dùng Qt offscreen; kiểm tra desktop X11 được ghi riêng bên dưới.
- VM dùng chung tài nguyên. Thời gian/RSS dưới đây là số quan sát của phiên,
  không phải benchmark phần cứng cô lập hoặc cam kết hiệu năng.

Số đo, từng ca CLI, kết quả desktop và SHA-256 source/build được lưu trong
[biên nhận JSON](evidence/cloud_stability_2026-11-27.json), tổng hợp từ ba probe
model, vòng lặp GUI và PyInstaller của phiên này.
Binary, fixture và DB tạm không được đưa vào repository.

## Lỗi đã tái hiện và sửa

| Lỗi | Sửa và kiểm thử hồi quy |
| --- | --- |
| Linux chấp nhận đường dẫn Windows không an toàn do dùng quy tắc của host | DB kiểm tra cả `PurePosixPath`/`PureWindowsPath`; artifact store từ chối drive/root/UNC, chuẩn hóa separator rồi `resolve` và kiểm tra containment. Xem [storage tests](../tests/test_storage_edges.py). |
| JSON chứa số nguyên hợp lệ nhưng lớn hơn float64 làm lộ `OverflowError` | Chuyển lỗi chuyển đổi về `ValueError` theo hợp đồng loader/CLI. Kiểm tra weights, mean, intercept, L2 và features trong [model stress tests](../tests/test_deepfake_model_stress.py). |
| Chạy lại sau khi hủy còn dòng trạng thái hủy của lần trước | Xóa trạng thái khi bắt đầu/kết thúc/lỗi; kiểm tra lại đúng nguồn ảnh, model và report. Xem [GUI lifecycle tests](../tests/test_ui_job_lifecycle.py). |
| Đóng cửa sổ khi worker còn chạy không chờ tác vụ kết thúc an toàn | Yêu cầu hủy, khóa tương tác và dùng timer chờ pool rảnh; event loop tiếp tục xử lý. Kiểm tra cả phân tích và tăng cường. |
| History dùng `currentRow` thay vì hàng đang chọn, có thể khóa nút hoặc trỏ sai report | Dùng selection thực, chỉ chấp nhận một hàng; kiểm tra chọn bằng accessibility, chuột, bàn phím và xóa selection trong [History selection tests](../tests/test_ui_history_selection.py). |

Tái hiện trước sửa storage: **6 failed, 85 passed** ở ba tệp persistence,
storage edges và enhancement study. Năm lỗi là validation cần nhất quán đa nền tảng.
Ca còn lại gọi `cmd /c mklink /J`: chỉ chạy trên Windows; ca symlink thật tương
đương chạy trên Linux và xác nhận thư mục nguồn không bị thay đổi.
Nhóm storage và tích hợp sau sửa: **197 passed, 1 skipped**; skip là junction Windows.

## Model và CLI

Biên nhận model ghi **41/41 tiến trình đạt**, không timeout hoặc traceback ngoài
hợp đồng. Các ca thất bại dự kiến được tính đạt khi trả đúng mã lỗi và giữ dữ liệu.

- Huấn luyện trong 5 tiến trình mới cho cùng SHA-256 model; đánh giá lặp 10 lần
  cho cùng đầu ra. Mẫu dev bị loại vẫn nằm trong ledger/báo cáo.
- 150 lần fit số học không lỗi; 5.000 vòng load/predict trong cùng tiến trình
  có cùng score, và 500 lần load model sai đều bị từ chối.
- Từ chối JSON hỏng, UTF-8 sai, file quá lớn, số không hữu hạn/số nguyên quá lớn,
  checksum/schema/variant/config sai và dữ liệu test trùng provenance dev.
- Input thiếu/hỏng/quá nhỏ và ảnh phẳng đi qua nhánh lỗi hoặc từ chối đo phù hợp.
- 32 file đầu vào được kiểm tra không đổi bytes; không tạo file input thiếu.
  Huấn luyện thất bại không để lại output dở dang; output cũ không bị ghi đè.

| Nhóm tiến trình | Số lần | Trung vị thời gian | RSS đỉnh lấy mẫu cao nhất |
| --- | ---: | ---: | ---: |
| Train | 5 | 0.538 s | 79.68 MiB |
| Evaluate | 10 | 0.414 s | 76.66 MiB |
| Analyze với model hợp lệ | 3 | 0.449 s | 93.86 MiB |

RSS tiến trình model sống lâu: 65436 → 66384 KiB sau GC, tăng 948 KiB.
Probe lấy mẫu `/proc` mỗi 10 ms có thể bỏ lỡ đỉnh ngắn; không diễn giải số này
thành chứng minh không rò bộ nhớ. Tập test tổng hợp có 8 mẫu, chỉ 3 đủ điều kiện;
coverage này kiểm tra cơ chế loại mẫu, không đại diện cho dữ liệu thực.

## GUI lặp tác vụ

Biên nhận offscreen ghi **45 vòng**: 30 hoàn tất, 15 hủy, 30 export Markdown;
thời gian tổng 5.28 s, `errors=[]`, số worker hoạt động cuối bằng 0.
Trường `rss_mb` của probe tăng từ 183.23 lên 206.25 ở vòng 45;
sau vòng 10 là 204.60.
Các số đo không khẳng định bộ nhớ luôn giữ nguyên trong mọi phiên sử dụng.

Kiểm thử trong repository dùng worker Qt thật để kiểm tra: chặn chạy chồng,
không đổi ảnh/model khi tác vụ đang chạy, report/export/history khớp nguồn,
giải phóng ảnh/worker cũ và đóng cửa sổ không chặn event loop.

## Bản đóng gói Linux

PyInstaller 6.22.3 dựng `ImageGuard.spec` thành công: **22.80 s**, peak child RSS
**159384 KiB**. Bundle có đúng một `haarcascade_frontalface_default.xml`
(930127 bytes), trùng OpenCV đã cài và nạp được bằng `CascadeClassifier`.

- Binary SHA-256: `deb9b87d80e512cff576d8b7da2ebf9e7128111e0796af70f4245bdd1c405d94`.
- Haar XML SHA-256: `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0`.
- Ba cảnh báo build liên quan driver DB tùy chọn; `sqlite3` của ứng dụng có trong bundle.
- `onnxruntime` chưa cài trong bản kiểm tra này; NAFNet/LaMa tùy chọn không khả dụng.
- `ldd` bootloader, Python và Qt6 XCB không báo thiếu thư viện trên máy build.
  Build thành công chưa chứng minh tương thích với mọi bản phân phối Linux.

## Kiểm tra cuối trên cloud

**Full suite sau tất cả bản sửa:** 2.030 passed, 13 skipped, 1 warning trong
100.53 s, exit code 0. Ruff toàn repository và `git diff --check` đạt.
Warning đến từ ca cố ý tạo header ảnh 100 MP để kiểm tra giới hạn trước decode.
Các bài kiểm tra cần PowerShell/junction Windows hoặc weight ONNX riêng không
thể được dùng làm bằng chứng đã chạy trên môi trường Linux này.

**Desktop X11 thật:** mở ứng dụng từ source và từ binary Linux vừa đóng gói;
chạy Deepfake mặc định và model đã học trên cùng ảnh control 320×320. Cả hai
hiển thị Haar backend khả dụng và kết quả model khớp nhau. Đường dẫn model có
dấu tiếng Việt nạp được. Source GUI từ chối JSON chứa số nguyên quá lớn, giữ
model hợp lệ đang chọn; PNG hỏng bị từ chối và không giữ kết quả của ảnh cũ.

Trên binary sau sửa, chọn dòng History bằng accessibility đã bật đúng các nút;
mở lại đúng report, đọc heatmap đã lưu và xuất Markdown thành công. Kiểm tra file
export xác nhận ID report và SHA-256 model đúng dòng được chọn. SQLite
`PRAGMA integrity_check` trả `ok`; log source và binary không có output lỗi.

Hủy, chạy lại và đóng cửa sổ khi worker còn hoạt động được kiểm tra bằng worker
Qt thật và event loop offscreen, gồm race lúc kết quả đã tới GUI nhưng runnable
chưa thoát. Không suy diễn các bài này thành đo độ trễ đóng cửa sổ trên mọi máy.
Chưa kiểm tra camera vật lý hoặc độ chính xác Deepfake trên ảnh thực có nhãn.

## Chạy lại và CI

Tại repository root trên Ubuntu 24.04, sau khi tạo và kích hoạt venv Python 3.13:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  libegl1 libgl1 libopengl0 libxkbcommon0 libdbus-1-3
python -m pip install -e '.[dev]'
export QT_QPA_PLATFORM=offscreen
python -m ruff check src tests scripts
python -m pytest -q
python -m pytest -q tests/test_deepfake_model_stress.py \
  tests/test_deepfake_training_cli.py tests/test_deepfake_model_integration.py \
  tests/test_ui_job_lifecycle.py tests/test_ui_history_selection.py \
  tests/test_storage_edges.py \
  tests/test_persistence.py tests/test_enhance_study.py
```

Job [`test-linux`](../.github/workflows/ci.yml) dùng Ubuntu 24.04/Python 3.13,
cài các thư viện Qt hệ thống, rồi Ruff và toàn bộ pytest offscreen; không
`continue-on-error`. Kết quả CI của commit đang xem có trong
[PR #242](https://github.com/JasonTM17/ImageGuard-Forensics/pull/242/checks).

Ví dụ CLI dưới đây cần manifest thật theo [giao thức đánh giá](deepfake-evaluation.md),
dev/test tách nhóm nguồn, cấu hình cố định và tên output chưa tồn tại:

```bash
python scripts/train_deepfake_model.py \
  --manifest data/deepfake/manifest.json --root data/deepfake \
  --config configs/thresholds.yaml --folds 5 --seed 20261127 --l2 1.0 \
  --output data/deepfake/dip_model.json --report data/deepfake/dev_training.json
python scripts/evaluate_deepfake_cues.py \
  --manifest data/deepfake/manifest.json --root data/deepfake --split test \
  --config configs/thresholds.yaml --model data/deepfake/dip_model.json \
  --output data/deepfake/test_model.json
python -m imageguard analyze data/anh_can_kiem_tra.jpg \
  --methods deepfake --config configs/thresholds.yaml \
  --deepfake-model data/deepfake/dip_model.json --no-persist
```

Chất lượng nhận diện vẫn cần được đo trên ảnh/video có nhãn độc lập, đủ đa dạng
và tách nguồn. Các kiểm tra ở đây không biến score thành xác suất Deepfake.
