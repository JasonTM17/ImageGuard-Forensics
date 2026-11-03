# Reading List — tài liệu nền cho đồ án

Chỉ liệt kê nguồn locate được (paper kinh điển, docs chính thức). Đọc theo
thứ tự — mỗi mục ghi "đọc để hiểu gì".

## Copy-move detection

1. **Fridrich, Soukal, Lukáš (2003)** — *Detection of Copy-Move Forgery in
   Digital Images* (DCT block matching + lexicographic sort gốc). Nền cho
   `forensics/block/`.
2. **Christlein et al. (2012)** — *An Evaluation of Popular Copy-Move
   Forgery Detection Approaches* — survey + benchmark, giải thích vì sao
   keypoint vs block trade-off khác nhau.
3. **Lowe (2004)** — *Distinctive Image Features from Scale-Invariant
   Keypoints* — SIFT gốc: scale space, 128-dim descriptor, ratio test.
   Đọc §6 (matching) để hiểu `ratio_rho` trong config.

## Compression artifacts

4. **Krawetz (2007)** — *A Picture's Worth…* (ELA, HFactor blog) — ý tưởng
   re-save difference; đọc kèm phần limitation (ELA yếu trên re-save).
5. **Popescu & Farid (2004)** — *Exposing Digital Forgeries by Detecting
   Traces of Resampling* — nền cho resampling/EM (đã cut scope, đọc để
   trình bày hướng mở rộng).

## Noise forensics

6. **Lukáš, Fridrich, Goljan (2006)** — *Detecting Digital Image Forgeries
   Using Sensor Pattern Noise* — PRNU; hiện tại app chỉ làm σ-residual
   đơn giản, paper này là hướng nâng cấp đúng.

## Đánh giá & dataset

7. **Tralic et al. (2013)** — *CoMoFoD — New Database for Copy-Move
   Forgery Detection* — dataset dùng trong `evaluation/`.
8. **Christlein et al. evaluation protocol** (cùng #2) — image-level vs
   pixel-level F1, lý do split-by-base-image.

## Tooling

9. OpenCV docs — `cv2.SIFT`, `findHomography` (RANSAC), `integral`.
10. PySide6 docs — `QThreadPool`/`QRunnable`, `QImage` ownership (lý do
    `.copy()` bắt buộc).
