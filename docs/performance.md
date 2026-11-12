# Hiệu năng runtime — benchmark phương pháp phân tích

Tài liệu ghi **thời gian wall-clock đo trên một máy**, tái lập qua `scripts/bench_methods.py`. Không phải claim độ chính xác phát hiện giả mạo.

## Lệnh đo (đã chạy)

Từ thư mục gốc worktree, Python 3.14 trong virtualenv dự án, `PYTHONPATH` trỏ `src`:

```powershell
$env:PYTHONPATH='src'
python scripts/bench_methods.py --out $env:TEMP\imageguard_bench_methods_full.json
```

- Seed ảnh synth: `424242` (mặc định script).
- Kích thước vuông: 256, 512, 1024, 2048 px.
- Lặp mỗi ô: `3` (mặc định `--repetitions`).
- ELA chạy trên file JPEG (quality 85); các method khác trên PNG cùng nội dung synth (`synthforge._base_image`).
- Đỉnh bộ nhớ: **NOT_RUN** (script cố ý bỏ qua — `tracemalloc` không đại diện cho OpenCV).

## Môi trường đo

| Mục | Giá trị |
| --- | --- |
| OS | Windows 11 |
| Machine | AMD64 |
| CPU (`platform.processor`) | Intel64 Family 6 Model 141 Stepping 1, GenuineIntel |
| Python | 3.14.2 |
| OpenCV | 5.0.0 |
| `cv2.getNumThreads()` lúc đo | 16 |

## Bảng `median_s` (giây) — sao chép từ JSON đầu ra

Nguồn: file JSON tạm `imageguard_bench_methods_full.json` (không nằm trong repo `data/`).

| method | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| keypoint | 0.028280600003199652 | 0.14604559999861522 | 1.6158136000012746 | 47.603493499998876 |
| block | 0.5125292000011541 | 2.1160541999997804 | 5.9247325000033015 | 6.8929092999969725 |
| ela | 0.0210538000028464 | 0.08680209999874933 | 0.356089000000793 | 1.477187500000582 |
| noise | 0.00232759999926202 | 0.010158300006878562 | 0.046210400003474206 | 0.14054319999559084 |
| noise_local | 0.0024193999997805804 | 0.007926899997983128 | 0.029004199997871183 | 0.10443579999991925 |
| freq | 0.0057077999954344705 | 0.026458199994522147 | 0.10603439999977127 | 0.4724768999949447 |
| splicing | 0.001950900004885625 | 0.009691999999631662 | 0.03542530000413535 | 0.1464432999928249 |
| **analysis_service (không persist)** | 0.5253770999988774 | 2.1750727000035113 | 6.245478799995908 | 51.22097900000517 |

`analysis_service` = `keypoint` + `block` + `ela` + EXIF audit, không ghi DB/artifact (`images`/`analyses` = `None`).

## Bảng `min_s` (giây) — cùng JSON

| method | 256 | 512 | 1024 | 2048 |
| --- | --- | --- | --- | --- |
| keypoint | 0.025657400001364294 | 0.14088679999986198 | 1.5793267999979435 | 42.4330076000042 |
| block | 0.45137040000554407 | 1.9114067999980762 | 5.003902600001311 | 5.79531809999753 |
| ela | 0.01985659999627387 | 0.08061649999581277 | 0.35571120000531664 | 1.4157370000029914 |
| noise | 0.0020080000031157397 | 0.00916480000159936 | 0.03932090000307653 | 0.13551359999837587 |
| noise_local | 0.002133200003299862 | 0.0075813000003108755 | 0.026988799996615853 | 0.10320089999731863 |
| freq | 0.005410599995229859 | 0.025110200003837235 | 0.10206490000564372 | 0.45270929999969667 |
| splicing | 0.0015839999978197739 | 0.008404499996686354 | 0.032647499996528495 | 0.14095880000240868 |
| **analysis_service (không persist)** | 0.514902399998391 | 2.140188899997156 | 5.967346199999156 | 47.452998200002185 |

## Nhận xét (chỉ từ bảng trên)

1. **Chiếm thời gian ở ảnh lớn:** tại 2048 px, `median_s` của `keypoint` (47.603493499998876 s) lớn hơn mọi method khác trong bảng; `block` đứng thứ hai (6.8929092999969725 s).
2. **Tỷ lệ theo kích thước (median_s, `keypoint`):** 256 → 512 ×5.16; 512 → 1024 ×11.06; 1024 → 2048 ×29.46 — tăng nhanh hơn khi cạnh ảnh lớn.
3. **Pipeline mặc định:** `analysis_service` tại 2048 px `median_s` 51.22097900000517 s, cùng bậc với tổng ba method mặc định trong đó `keypoint` chiếm phần lớn ở size này.
4. **Method phụ trợ** (`noise`, `noise_local`, `splicing`, `freq`): `median_s` tại 2048 px đều dưới 0.5 s trong lần đo này — nhỏ hơn `keypoint` và `block` tại cùng size.

## Nguyên nhân (profiling một lần, cùng máy)

Profiling bằng `cProfile` và `perf_counter` trên ảnh synth 2048 px cho thấy phần lớn thời gian của `keypoint` nằm ở `cv2.BFMatcher.knnMatch` trong `self_match` (so khớp brute-force từng descriptor với mọi descriptor khác), không nằm ở RANSAC hay dựng mask. Số keypoint SIFT tăng xấp xỉ theo diện tích ảnh (`nfeatures: 0`, không giới hạn) nên chi phí so khớp tăng nhanh hơn diện tích. Ảnh phẳng ít keypoint (dưới `k_min`) thoát sớm và rất nhanh; ảnh nhiều texture là trường hợp chậm nhất. Mọi cách giảm thời gian bằng hạ kích thước làm việc hoặc giới hạn số keypoint đều **đổi đầu ra của detector** và cần quyết định riêng; tài liệu này không đề xuất đổi ngưỡng.

## Hạn chế

- Một máy, một phiên chạy; không đối chứng phần cứng khác.
- Trạng thái cache CPU/OS không cố định giữa các lần lặp (ấm sau lần đầu trong cùng phiên).
- OpenCV dùng 16 thread (`cv2.getNumThreads()`); thay đổi thread pool có thể đổi thời gian.
- Ảnh synth có texture (không phải ảnh thật); thời gian trên file JPEG/PNG thực tế có thể khác.
- Smoke CI: `pytest tests/test_bench_methods_smoke.py` với `--quick` / size nhỏ — chỉ kiểm cấu trúc JSON, **không** so sánh số giây.
