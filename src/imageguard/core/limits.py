"""Giới hạn cứng chống input độc hại. Mọi con số phải có tên + lý do,
override được qua configs/thresholds.yaml (mục `limits`)."""

MAX_FILE_BYTES = 100 * 1024 * 1024  # 100MB — chặn file bất thường trước decode
MAX_PIXELS = 80_000_000  # 80MP — chống decompression bomb (px khai trong header)
# Ghi chú bộ nhớ: aux methods (noise_map/noise_local/freq_view) đọc ảnh full-res
# và cần ~10-25 byte/px (integral float64, FFT complex128) → 80MP có thể >2GB.
# AUX_MAX_SIDE downscale trước khi phân tích (kernel đổi tỉ lệ thuận, mask/heat
# upscale về kích thước gốc) — giống pattern WORK_MAX_SIDE của detector.
MIN_SIDE = 64  # px — dưới ngưỡng này không đủ tín hiệu forensic → INCONCLUSIVE
WORK_MAX_SIDE = 2048  # px — working copy tối đa đưa vào detector, ghi scale_factor
AUX_MAX_SIDE = 4096  # px — working copy tối đa cho aux methods (noise/freq)
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
