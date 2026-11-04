"""Giới hạn cứng chống input độc hại. Mọi con số phải có tên + lý do,
override được qua configs/thresholds.yaml (mục `limits`)."""

MAX_FILE_BYTES = 100 * 1024 * 1024  # 100MB — chặn file bất thường trước decode
MAX_PIXELS = 80_000_000  # 80MP — chống decompression bomb (px khai trong header)
# Ghi chú bộ nhớ: aux methods (noise_map/noise_local/freq_view/ELA) đọc doc.array
# ở FULL RESOLUTION — 80MP có thể cần ~2GB+ RAM cho integral/FFT. WORK_MAX_SIDE
# (2048) chỉ áp dụng cho preprocess_keypoint. Nếu OOM với ảnh lớn, giảm max_pixels.
MIN_SIDE = 64  # px — dưới ngưỡng này không đủ tín hiệu forensic → INCONCLUSIVE
WORK_MAX_SIDE = 2048  # px — working copy tối đa đưa vào detector, ghi scale_factor
MAX_DECODE_SIDE = 16384  # px — hard ceiling cạnh đơn, phòng lỗi header khai sai

SNIFF_BYTES = 16  # số byte đầu đọc để nhận diện magic

MAGIC_TABLE: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"RIFF", "webp"),  # kiểm thêm 'WEBP' ở offset 8
]
