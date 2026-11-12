# Kịch bản demo trực tiếp — ImageGuard Forensics (8–10 phút)

Tài liệu cho **bảo vệ đồ án** môn Xử lý ảnh: người trình bày chưa từng mở repo vẫn có thể chạy theo từng bước trên **Windows PowerShell**. Mọi lệnh dưới đây đã được chạy thử trong thư mục scratch `<scratch>` (tách khỏi repo); đường dẫn máy thật được thay bằng placeholder.

**Ranh giới khoa học (nói trước hội đồng):** Ứng dụng chỉ báo *dấu hiệu nghi vấn* — không khẳng định ảnh giả, không chứng minh ảnh nguyên bản ([`README.md`](../README.md), [`limitations.md`](limitations.md)).

| Placeholder | Ý nghĩa |
| --- | --- |
| `<repo>` | Thư mục clone ImageGuard Forensics (worktree đồ án) |
| `<python>` | `Python.exe` trong venv đã cài `-e ".[dev]"` |
| `<scratch>` | Thư mục tạm ngoài repo (ví dụ tạo `mkdir <scratch>` rồi `cd` vào đó) |

**Chuẩn bị (trước buổi bảo vệ):** Cài dependency theo [`README.md`](../README.md). Không chạy `scripts\verify_env.py` trong buổi demo — script đó **ghi đè** `docs/environment.md`. Thay vào đó dùng bước kiểm tra môi trường ở Phần 1.

**Ma trận đã chạy thử (ghi nhận 2026-11-12):**

| Bước | Trạng thái ghi nhận |
| --- | --- |
| Kiểm tra phiên bản / import | **PASS** |
| Sinh ảnh synthforge (CLI hoặc Python) | **PASS** |
| `imageguard analyze` (CLI) | **PASS** (~15,7 s cho ảnh 512×512, keypoint+block) |
| Xuất Markdown qua `ReportService` (tương đương nút GUI) | **PASS** |
| Smoke GUI 4 s (process không thoát sớm) | **PASS** |
| Demo `ipkit` (CLI) | **PASS** |
| Thao tác GUI có màn hình (overlay, lịch sử, Tăng cường) | **NOT_RUN** trong phiên tự động — presenter chạy tay khi máy chiếu |

---

## Phần 0 — Mở đầu (≈30 giây)

1. Giới thiệu: desktop PySide6, engine forensics tách khỏi UI, verdict thận trọng (`INCONCLUSIVE` là câu trả lời hợp lệ).
2. Luồng demo: ảnh tổng hợp có mask GT → CLI JSON → GUI overlay → lịch sử/thống kê → xuất báo cáo → tab Tăng cường (xem, không phân tích) → giới hạn đã biết.

---

## Phần 1 — Kiểm tra môi trường (≈1 phút)

```powershell
cd <repo>
$env:PYTHONPATH = "<repo>\src"
<python> --version
<python> -c "import cv2, numpy, PySide6; print('opencv', cv2.__version__); print('numpy', numpy.__version__); print('PySide6', PySide6.__version__); print('SIFT', hasattr(cv2,'SIFT_create'))"
<python> -m imageguard --version
```

**Đầu ra mẫu (đã quan sát khi chạy thử):**

```text
Python 3.14.2
opencv 5.0.0
numpy 2.5.4
PySide6 6.12.0
SIFT True
ImageGuard Forensics v0.1.0
```

Nếu `SIFT False` hoặc thiếu PySide6 → dừng demo, cài lại `.[dev]` theo README.

---

## Phần 2 — Sinh ảnh copy-move tổng hợp (≈1,5 phút)

Dùng generator nội bộ **synthforge** (deterministic theo seed; có `*_O`, `*_F`, `*_M`, `meta.json`). Chiến lược dataset: [`docs/dataset-strategy.md`](dataset-strategy.md).

```powershell
cd <scratch>
$env:PYTHONPATH = "<repo>\src"
New-Item -ItemType Directory -Force -Path .\synth | Out-Null
<python> -m imageguard.evaluation.synthforge --n 1 --seed 20261112 --out .\synth --tier easy
Get-ChildItem .\synth
```

**Đầu ra mẫu:**

```text
Wrote 1 samples to .\synth (manifest: .\synth\meta.json)

easy_0000_F.png   # ảnh chỉnh sửa tổng hợp — dùng cho phân tích
easy_0000_O.png   # gốc
easy_0000_M.png   # mask GT (255 = vùng copy-move)
meta.json
```

Kích thước file quan sát một lần chạy: `easy_0000_F.png` ≈ 545 607 byte; `easy_0000_O.png` ≈ 558 474 byte; `easy_0000_M.png` ≈ 1 545 byte; ảnh **512×512**.

**Phương án B (một file, API như `scripts/make_report_figures.py`):**

```powershell
cd <scratch>
$env:PYTHONPATH = "<repo>\src"
<python> -c @"
import cv2
from pathlib import Path
import numpy as np
from imageguard.evaluation.synthforge import ForgeSpec, _base_image, forge
rng = np.random.default_rng(20261112)
base = _base_image(rng, (512, 512))
spec = ForgeSpec((512,512),'rect',[[120,120],[280,120],[280,280],[120,280]],(150,150),0.0,1.0,'none','easy')
forged, mask = forge(base, spec)
out = Path('single_demo.png')
cv2.imwrite(str(out), forged)
print('wrote', out.name, 'bytes', out.stat().st_size, 'mask_pixels', int((mask>0).sum()))
"@
```

Đầu ra mẫu: `single_demo.png` ≈ 558 778 byte, `mask_pixels` ≈ 25 921.

**Câu nói cho hội đồng:** Ảnh không phải ảnh cá nhân; mask GT chỉ phục vụ đánh giá nội bộ, không xuất hiện trong verdict GUI.

---

## Phần 3 — CLI `analyze` (≈2 phút)

Phân tích file ảnh chỉnh sửa tổng hợp; in JSON `ForensicReport` ra stdout ([`docs/user-guide.md`](user-guide.md)).

```powershell
cd <scratch>
$env:PYTHONPATH = "<repo>\src"
$env:PYTHONIOENCODING = "utf-8"
$img = (Resolve-Path .\synth\easy_0000_F.png).Path
<python> -m imageguard analyze $img --methods keypoint,block --no-persist | Select-Object -First 12
```

**Kết quả đã quan sát trên cùng ảnh synth:**

| Trường | Giá trị |
| --- | --- |
| `image_width` × `image_height` | 512 × 512 |
| `overall_verdict` | `suspicious_indication` |
| `copy_move_keypoint` | `suspicious_indication`, `has_mask`: true |
| `copy_move_block` | `suspicious_indication`, `has_mask`: true |
| `exif_audit` | `inconclusive` (cue phụ; không tự kết luận) |

Thời gian wall-clock một lần đo: **≈15,7 s** (máy chạy thử demo).

**Câu nói:** `suspicious_indication` = có cấu trúc giống copy-move trong pipeline, **không** đồng nghĩa “ảnh giả”. Exit code ≠ 0 chỉ khi lỗi đọc file (`ErrorResult`), không phải verdict.

**Tùy chọn — lưu DB trong `<scratch>` (phục vụ tab Lịch sử GUI):**

```powershell
cd <scratch>
Remove-Item -Recurse -Force .\data -ErrorAction SilentlyContinue
<python> -m imageguard analyze $img --methods keypoint,block
Get-ChildItem .\data\artifacts -Recurse -Filter *.png
```

Quan sát: thư mục `data\artifacts\<analysis_id>\` chứa `copy_move_keypoint_mask.png`, `copy_move_block_mask.png`, heatmap, `report.json`.

---

## Phần 4 — GUI: mở ảnh, phân tích, overlay (≈2,5 phút) — thao tác trực tiếp

> **Trạng thái ghi nhận:** **NOT_RUN** (cần màn hình). Đã **PASS** smoke: process `python -m imageguard <ảnh>` chạy ≥4 s không thoát sớm (offscreen).

```powershell
cd <repo>
# Tùy chọn: trỏ DB và artifact vào <scratch> nếu vừa phân tích ở Phần 3
# (mặc định app dùng <repo>\data khi cwd là <repo> — presenter chọn một: chạy lại Phân tích trong GUI hoặc copy/link data)
<python> -m imageguard
# hoặc mở thẳng file:
<python> -m imageguard <scratch>\synth\easy_0000_F.png
```

**Checklist trên máy chiếu (theo [`docs/ui-design.md`](ui-design.md), [`docs/user-guide.md`](user-guide.md)):**

1. Tab **Phân tích** — **Mở ảnh** → chọn `easy_0000_F.png`.
2. Bật **Copy-Move keypoint** và **Copy-Move block** → **Phân tích**; chờ progress (dự trù ~15–30 s cho 512²).
3. Đọc banner verdict (cam / xám / vàng) — nhắc disclaimer dưới cùng.
4. Dropdown **overlay**: mask đỏ và/hoặc heatmap; zoom (lăn chuột), pan (kéo). So sánh **cảm quan** với `easy_0000_M.png` (GT) nếu mở song song — **không** gọi đó là “độ chính xác %”.
5. Thẻ từng method: đọc `score_definition` và `limitations` trên card.

---

## Phần 5 — Lịch sử, Thống kê, xuất Markdown (≈1,5 phút)

### 5a. GUI (NOT_RUN tự động)

1. Tab **Lịch sử** — chọn dòng vừa phân tích → mở lại kết quả.
2. Tab **Thống kê** — tổng hợp phiên (số lần chạy, verdict) từ SQLite.
3. Nút **Xuất báo cáo (Markdown)** → lưu `bao-cao.md`; mở file, chỉ vào disclaimer và bảng method — **không** có xác suất “ảnh giả”.

### 5b. Dự phòng đã chạy thử (PASS) — khi cần chứng minh export trước hội đồng mà chưa kịp bấm GUI

Sau Phần 3 (persist), trong `<scratch>`:

```powershell
cd <scratch>
$env:PYTHONPATH = "<repo>\src"
<python> -c @"
from pathlib import Path
from imageguard.storage.artifacts import ArtifactStore
from imageguard.services.report_service import ReportService
store = ArtifactStore(Path('data/artifacts'))
aid = next(p.name for p in Path('data/artifacts').iterdir() if p.is_dir())
out = Path('demo_report.md')
ReportService(store).export_markdown(aid, out)
print('wrote', out, 'lines', len(out.read_text(encoding='utf-8').splitlines()))
"@
Get-Content .\demo_report.md -TotalCount 15
```

Đầu ra mẫu: file **60 dòng**, dòng đầu `# Báo cáo phân tích — easy_0000_F.png`, block trích dẫn disclaimer tiếng Việt.

---

## Phần 6 — Tab Tăng cường + `ipkit` (≈1,5 phút)

### 6a. Tab **Tăng cường** (NOT_RUN tự động)

Theo [`docs/user-guide.md`](user-guide.md): operator **CLAHE**, **Unsharp mask**, **NLM** — chỉ **xem**, không gọi `AnalysisService`, không ghi verdict.

1. Vẫn mở ảnh chỉnh sửa tổng hợp (hoặc ảnh camera minh họa).
2. Tab **Tăng cường** → chọn **CLAHE** (histogram cục bộ — liên hệ môn học) → **Tăng cường**.
3. Đọc banner cố định: bản xem, không phải bằng chứng.
4. (Tùy chọn) Thanh trộn 0–100; **không** chạy lại Phân tích trên ảnh đã tăng cường.

**NAFNet (AI):** chỉ demo nếu đã cài extra `lama` và đủ ONNX — không bắt buộc; xem [`docs/enhancement-study.md`](enhancement-study.md).

### 6b. Toolkit `ipkit` (PASS — CLI, ≈30 giây)

`imageguard.ipkit` là toán tử học tập; **không** nối vào pipeline verdict ([`docs/image-processing-toolkit.md`](image-processing-toolkit.md)). Demo nhanh trên canvas phẳng:

```powershell
$env:PYTHONPATH = "<repo>\src"
<python> -c "import numpy as np; from imageguard.ipkit import sobel, gradient_magnitude; g=np.full((64,64),128,np.uint8); gx,gy=sobel(g); print('shape', gx.shape, 'grad_max', float(gradient_magnitude(g).max()))"
```

Đầu ra mẫu: `shape (64, 64) grad_max 0.0` (ảnh phẳng → gradient 0 — minh họa đúng toán).

**Câu nói:** Tab Tăng cường và `ipkit` phục vụ **hiển thị / học thuật toán xử lý ảnh**; detector copy-move chỉ chạy trên file gốc người dùng mở ở tab Phân tích.

---

## Phần 7 — Giới hạn & kết (≈1 phút)

Trích từ [`limitations.md`](limitations.md) — **không bổ sung con số mới** trong lời nói:

| Chủ đề | Điểm nhấn khi trả lời |
| --- | --- |
| Phạm vi số liệu CASIA 103 cặp | Chỉ minh họa metric trên cohort đã lọc; không generalize sang ảnh tùy ý |
| Keypoint trên ảnh gốc | Nhiều `INCONCLUSIVE` (59/103) — abstain có chủ đích |
| Block trên ảnh gốc | 0 FP suspicious trên mẫu đó — README nhấn **không** suy ra FPR luôn 0 |
| Verdict fusion GUI | Chưa đánh giá trong sự kiện đo từng method đơn lẻ |
| ELA | Chỉ có ý nghĩa mạnh với JPEG |
| Enhance / AI | Không đưa pixel enhanced vào fusion |

**Kết demo:** Mở `limitations.md` hoặc README → cảm ơn → nhận câu hỏi (xem Phụ lục B).

---

## Phụ lục A — Dòng thời gian gợi ý (tổng 8–10 phút)

| Phút | Nội dung |
| --- | --- |
| 0:00–0:30 | Mở đầu + ranh giới khoa học |
| 0:30–1:30 | Phần 1–2: env + synthforge |
| 1:30–3:30 | Phần 3: CLI JSON + giải thích verdict |
| 3:30–6:00 | Phần 4: GUI overlay + method cards |
| 6:00–7:30 | Phần 5: Lịch sử / Thống kê / xuất MD |
| 7:30–8:30 | Phần 6: CLAHE + ipkit |
| 8:30–10:00 | Phần 7: limitations + Q&A |

---

## Phụ lục B — Gợi ý trả lời hội đồng (10 câu)

Câu trả lời ngắn, bám [`limitations.md`](limitations.md) và [`README.md`](../README.md).

1. **Ứng dụng có kết luận ảnh giả không?**  
   Không. Chỉ báo *dấu hiệu nghi vấn* (`suspicious_indication`), *không thấy dấu hiệu*, hoặc *không đủ bằng chứng* (`INCONCLUSIVE`). Disclaimer cố định trên UI và trong báo cáo Markdown.

2. **`INCONCLUSIVE` có phải lỗi không?**  
   Không. Là abstain khi texture kém, tín hiệu mâu thuẫn, hoặc cue phụ không đủ — hành vi có chủ đích (keypoint trên ảnh gốc CASIA có nhiều inconclusive).

3. **Số % chính xác trên ảnh demo hôm nay?**  
   Không báo % trên ảnh live. Số trong README là trên cohort **103 cặp CASIA đã lọc** với oracle mask — không chứng minh cho ảnh camera bất kỳ.

4. **Keypoint hay Block tốt hơn?**  
   Trade-off đã ghi: keypoint recall cao hơn trên cohort nhưng nhiều inconclusive/FP trên gốc; block conservative hơn trên gốc (0 FP suspicious trong 103) nhưng miss nhiều ảnh chỉnh. Fusion trên GUI **chưa** được đánh giá trong sự kiện đo đơn method.

5. **ELA dùng cho PNG được không?**  
   ELA gắn JPEG; PNG/BMP thường “không áp dụng” hoặc không đứng một mình — xem [`docs/algorithms/comparison.md`](algorithms/comparison.md).

6. **Tab Tăng cường / NAFNet có làm ảnh “sạch” để qua detector không?**  
   Không. Enhance chỉ xem; không gọi `AnalysisService`, không đổi verdict. NAFNet là restorer xem, không phải detector giả mạo.

7. **`ipkit` có ảnh hưởng kết quả phân tích không?**  
   Không. Module tách biệt, không gọi từ detector ([`docs/image-processing-toolkit.md`](image-processing-toolkit.md)).

8. **Dữ liệu demo có vi phạm riêng tư không?**  
   Không. synthforge sinh ảnh procedural trong repo; không dùng ảnh cá nhân ([`docs/report/figures.md`](report/figures.md)).

9. **Giới hạn copy-move xoay/phóng to?**  
   Block DCT yếu khi biến đổi hình học lớn; tier `hard` synthforge có rotate/scale — có thể `NO_INDICATION` dù có forgery.

10. **Tại sao không ship `.exe` làm đường chính?**  
    [`docs/user-guide.md`](user-guide.md) và [`docs/packaging.md`](packaging.md): build PyInstaller từng gặp lỗi Qt DLL; demo khuyến nghị chạy source `python -m imageguard`.

---

## Phụ lục C — Xử lý sự cố nhanh

| Triệu chứng | Hướng xử lý |
| --- | --- |
| JSON CLI lỗi Unicode trên console | `$env:PYTHONIOENCODING = "utf-8"` |
| `ModuleNotFoundError: imageguard` | Đặt `$env:PYTHONPATH = "<repo>\src"` hoặc `pip install -e ".[dev]"` từ `<repo>` |
| Phân tích quá lâu trên máy yếu | Dùng `--methods keypoint` only hoặc ảnh 256² (`generate_set` size tùy chỉnh qua Python API) |
| GUI không lên | Kiểm tra PySide6; thử `run_app.bat`; xem stderr |

---

*Tài liệu đồ án — chỉ mô tả lệnh đã chạy thử; cập nhật khi đổi phiên bản tool hoặc API synthforge.*
