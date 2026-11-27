# Đánh giá dấu hiệu Deepfake cho đồ án Xử lí ảnh

## Phạm vi cần trình bày đúng

`deepfake_cues` hiện là bộ đo **dấu hiệu bất thường trên vùng mặt** bằng xử lí
ảnh cổ điển. Nó không chứa trọng số của một mạng đã được huấn luyện phân loại
Deepfake, không xác thực danh tính và không chứng minh ảnh thật hay giả.

Điểm tổng hợp là chỉ số heuristic từ nhiễu phần dư, texture, phổ và biên. Giá trị
`0.8` không có nghĩa là “80% Deepfake”. Ảnh da mịn, trang điểm, ánh sáng lệch,
khử nhiễu của camera, nền có nhiều chi tiết và ảnh nén đều cần được đưa vào
tập đối chứng khi đánh giá. Không nên tăng/giảm ngưỡng chỉ để một ảnh demo
hiện kết quả mong muốn.

Hợp đồng kết quả sau sửa:

- `cue_flag=true`: có dấu hiệu vượt điều kiện sàng lọc; verdict vẫn là
  `INCONCLUSIVE`, cần kiểm tra thêm.
- `cue_flag=false` và `cue_unavailable=false`: có đủ điều kiện đo nhưng chưa
  bật cờ; `NO_INDICATION` không đồng nghĩa ảnh thật.
- `cue_unavailable=true`, `score=null`: thiếu điều kiện đánh giá, ví dụ không
  tìm được mặt, ROI quá nhỏ hoặc thiếu nền đối chiếu. Trường
  `unavailable_reason` giải thích lý do.

Khi dùng mốc cùng phiên, cả ROI tham chiếu và ROI nghi vấn được kiểm tra
kích thước/tương phản trên bản làm việc **trước khi resize**. Mốc nhỏ hoặc
thiếu tương phản có `ref_status=quality_unavailable` và không được dùng để
nâng verdict. Các phép đo không áp dụng có giá trị `null`.

Không dùng `verdict == INCONCLUSIVE` làm nhãn “giả”: trạng thái này bao gồm cả
kết quả có dấu hiệu lẫn trường hợp thiếu dữ liệu. Script đánh giá chỉ đọc
`cue_flag` rõ ràng và chỉ tính mẫu đủ điều kiện.

## Giao thức thực nghiệm đề xuất

1. Chọn ảnh/video có nhãn gốc độc lập với thuật toán. Có thể dùng phần dữ liệu
   được cấp phép của FaceForensics++ hoặc bộ dữ liệu tương đương, bổ sung
   ảnh camera thật do nhóm tự thu nhận có sự đồng ý của người xuất hiện.
2. Tách `dev` và `test` theo **nhóm nguồn** trước khi chỉnh tham số. Một video
   gốc, các frame của nó, bản giả phát sinh, phiên bản nén/resize và các nguồn
   donor liên quan phải ở cùng nhóm. Không chia ngẫu nhiên các frame của cùng
   video sang cả hai tập. Nếu đánh giá khả năng tổng quát sang người mới,
   phải nhóm theo danh tính nữa; script chỉ có thể kiểm tra nhóm được khai báo.
3. Trên `dev`, xác định điều kiện ảnh đủ chất lượng và chọn ngưỡng. Lưu YAML,
   manifest và commit mã nguồn. **Đóng băng cấu hình trước khi chạy `test`.**
   Script không tự chọn ngưỡng và không có vòng tìm ngưỡng trên test.
4. Báo cáo riêng ảnh nguyên bản, JPEG/nén video, resize, thiếu sáng, mờ chuyển
   động và hậu cảnh khác nhau. Giữ cả mẫu lỗi đọc, không thấy mặt và chất lượng
   không đủ trong tổng số mẫu; đừng loại chúng âm thầm để làm đẹp kết quả.
5. Sau đánh giá cùng nguồn, thử một bộ dữ liệu/camera khác và giữ nguyên cấu
   hình. Đây là phép đo chuyển miền mới, không suy ra từ điểm trên tập cũ.

FaceForensics++ đưa nén và kích thước ảnh vào thiết kế đánh giá các phương
pháp chỉnh sửa khuôn mặt [1]. DeepfakeBench nhấn mạnh sự thống nhất của dữ
liệu đầu vào, điều kiện thí nghiệm và metric để tránh so sánh sai [2]. Các
bước trên là giao thức đề xuất cho dự án, không phải kết quả đã đo trên các
bộ dữ liệu đó.

## Manifest và cách chạy

Lưu ảnh ở thư mục dữ liệu ngoài Git. Manifest JSON dùng đường dẫn tương đối
với `--root`; không chứa URL tải ảnh hay đường dẫn tuyệt đối. Ví dụ cấu trúc
(cần thay các đường dẫn bằng ảnh đã chuẩn bị, đây không phải dữ liệu benchmark):

```json
{
  "version": 1,
  "samples": [
    {"id": "dev_real_001", "path": "dev/real_001.png", "label": "real", "split": "dev", "source_group": "source_001", "condition": "original"},
    {"id": "dev_fake_001", "path": "dev/fake_001.png", "label": "fake", "split": "dev", "source_group": "source_001", "condition": "original"},
    {"id": "test_real_002", "path": "test/real_002.jpg", "label": "real", "split": "test", "source_group": "source_002", "condition": "jpeg_q80"},
    {"id": "test_fake_002", "path": "test/fake_002.jpg", "label": "fake", "split": "test", "source_group": "source_002", "condition": "jpeg_q80"}
  ]
}
```

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_deepfake_cues.py `
  --manifest data\deepfake\manifest.json `
  --root data\deepfake `
  --split test `
  --config configs\thresholds.yaml `
  --output data\deepfake\test_cues.json
```

Trên Linux/macOS dùng `.venv/bin/python` và dấu `/`. Nếu bỏ `--output`, JSON
được in ra stdout. File kết quả có sẵn không bị ghi đè. Script đọc ảnh qua
`safe_open`, giữ giới hạn kích thước/định dạng của ứng dụng và không sửa ảnh gốc.

Script này đo **ảnh đơn và vùng mặt do detector chọn**. PNG được đánh dấu thu
nhận hai camera bị loại với lý do `unsupported_capture_layout`; ảnh rộng
thông thường không tự được coi là hai camera. Script không đánh giá toàn bộ các mặt trong ảnh nhóm,
đồng bộ hai cảm biến hay tín hiệu temporal của video. Muốn đánh giá video,
cần giao thức lấy frame và tổng hợp theo video riêng.

## Đọc kết quả

| Trường JSON | Ý nghĩa |
| --- | --- |
| `n_total`, `n_eligible`, `n_excluded` | Tổng mẫu, mẫu đủ điều kiện và mẫu bị từ chối đo |
| `coverage` | `n_eligible / n_total`; báo cả tổng thể và riêng từng lớp |
| `by_class.real.cue_rate_eligible` | Tỷ lệ ảnh thật đủ điều kiện bị bật cờ; false-cue rate trên mẫu đã đo |
| `by_class.fake.cue_rate_eligible` | Tỷ lệ ảnh giả đủ điều kiện được bật cờ; cue recall trên mẫu đã đo |
| `by_class.*.cue_yield_all` | Số mẫu bật cờ / toàn bộ mẫu của lớp đó, kể cả mẫu bị loại ở mẫu số |
| `score_roc_auc_eligible` | Khả năng xếp hạng điểm ảnh giả cao hơn ảnh thật trong mẫu đủ điều kiện; bằng nhau tính nửa điểm |
| `exclusions` | Số mẫu theo từng lý do thiếu dữ liệu/lỗi, không bị tính thành âm tính đúng |
| `by_condition` | Cùng các metric trên cho từng điều kiện ảnh được khai báo |
| `manifest_sha256`, `config_sha256`, `config` | Dấu vết manifest/cấu hình và snapshot để tái hiện |
| `samples[].sha256` | SHA-256 của bytes ảnh; đối chiếu với dữ liệu đang lưu |
| `integrity.unhashed_ids` | Mẫu không đọc/hash được hoặc vượt giới hạn bytes ở bất kỳ split nào |

Khi một lớp không có mẫu đủ điều kiện, tỷ lệ/AUC tương ứng là `null`, không
phải 0 hay 1. AUC chỉ đánh giá thứ tự điểm, không làm điểm trở thành xác suất
đã hiệu chuẩn. Coverage thấp hoặc lệch mạnh giữa thật/giả khiến chỉ số trên
mẫu đủ điều kiện dễ gây hiểu lầm; cần trình bày chúng cùng nhau.

Trước khi chạy detector, script kiểm tra trùng ID, đường dẫn, `source_group`
chéo split và **trùng bytes ảnh trên toàn bộ manifest**, kể cả giữa `dev`
và `test`. Nó không phát hiện được mọi ảnh gần trùng sau nén/cắt/resize hay
hai video cùng một người; nhóm vẫn phải khai báo nguồn chính xác. Các mẫu
không hash được được liệt kê và cần giải quyết trước khi coi đây là một
benchmark hoàn chỉnh.

Các metric là ở cấp **frame/ảnh**, không coi nhiều frame cùng video là các
quan sát thống kê độc lập. Script chưa tính khoảng tin cậy theo nhóm, không
được dùng số frame để tự suy ra độ tin cậy của kết luận ở cấp video.

## Đối chứng số học cho bản sửa FFT (2026-11-27)

Đây là đối chứng tổng hợp của **một thành phần phổ**, không phải độ chính
xác phát hiện Deepfake. So mã gốc `21e3a6643468228a5cfc28556b2d1b093b53851a`
với bản sửa: mỗi kích thước có 50 seed `4567..4616`, nhiễu Gaussian σ=10,
lưới cosine hai chiều chu kỳ 3 px, biên độ 22. Điểm phổ nằm trong [0, 1].

| ROI | Điểm nhiễu TB trước | Điểm nhiễu TB sau | Điểm lưới trung vị trước | Điểm lưới trung vị sau |
| --- | ---: | ---: | ---: | ---: |
| 32×32 | 0.283165 | 0.000051 | 0.595025 | 0.882209 |
| 48×48 | 0.104710 | 0.000000 | 0.674677 | 1.000000 |
| 64×64 | 0.037918 | 0.000000 | 0.836556 | 1.000000 |
| 96×96 | 0.000000 | 0.000000 | 0.552357 | 1.000000 |
| 128×128 | 0.000000 | 0.000000 | 0.275316 | 1.000000 |
| 256×256 | 0.000000 | 0.000000 | 0.103114 | 1.000000 |

Trước sửa, nhiễu ở ROI 32 px có điểm khoảng 0,28 chỉ vì cách lấy top-8
bin; đồng thời lưới rõ ở ROI lớn có thể bị phân vị 99,8% bỏ qua và bị hạ
điểm. Bản sửa dùng tỷ lệ bin cố định và năng lượng phổ để xử lí hai lỗi đó.
Điểm 1 ở cột lưới chỉ là thành phần tuần hoàn bão hòa, không phải chắc chắn
Deepfake. Mặt nhỏ hơn ngưỡng mặc định 64 px vẫn không được cấp điểm tổng.

Kiểm thử hồi quy dùng seed khác và nhiều chu kỳ 3/4/5 px, gồm ROI lẻ/hình
chữ nhật, tại `tests/test_deepfake_spectral_validity.py`. Chạy:

```bash
.venv/bin/python -m pytest -q tests/test_deepfake_spectral_validity.py
```

## Nội dung nên đưa vào báo cáo và buổi bảo vệ

- Trình bày vai trò của từng phép xử lí ảnh: định vị ROI, median residual,
  LBP, FFT, Canny/Hough và bản đồ vùng bất thường.
- Đưa cả ví dụ phát hiện đúng, báo động trên ảnh thật, bỏ sót ảnh giả và
  trường hợp từ chối đo. Giải thích vì sao ảnh thật có thể tạo cùng dấu hiệu.
- Tách kiểm thử phần mềm (hợp đồng, hủy tác vụ, tránh chia nhầm camera,
  không hiển thị score khi thiếu mặt) khỏi đánh giá chất lượng nhận diện
  trên dữ liệu có nhãn. Test chạy xanh không chứng minh detector chính xác.
- Nếu mở rộng sang mạng học sâu, cần ghi model/weights, nguồn trọng số,
  giấy phép, tiền xử lí, thứ tự nhãn và đánh giá chuyển miền. Giữ phần Xử lí
  ảnh hiện tại làm nhánh giải thích/đối chứng; không ghép tùy ý một xác suất
  của mạng với điểm heuristic để tạo “độ tin cậy” mới.

**Hiện chưa kèm tập Deepfake có nhãn hay kết quả nhận diện thực nghiệm trong
bản sửa này.** Kiểm thử dùng ảnh sinh tổng hợp chỉ kiểm tra luồng phần mềm
và tính đúng của metric, không đại diện cho chất lượng nhận diện thực tế.

## Tài liệu tham khảo

1. Rössler và cộng sự, *FaceForensics++: Learning to Detect Manipulated
   Facial Images*, ICCV 2019.
   <https://openaccess.thecvf.com/content_ICCV_2019/html/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.html>
2. Yan và cộng sự, *DeepfakeBench: A Comprehensive Benchmark of Deepfake
   Detection*, NeurIPS 2023 Datasets and Benchmarks.
   <https://proceedings.neurips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html>
